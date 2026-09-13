/**
 * OpenAL cross platform audio library
 * Web Audio backend for Emscripten builds (ufront).
 *
 * This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Library General Public
 *  License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 *  License along with this library; if not, write to the
 *  Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 * Or go to http://www.gnu.org/copyleft/lgpl.html
 */

/* HOW IT WORKS
 *
 *   app thread ── AL calls ──▶ openal-soft (same wasm module, no proxying)
 *   mixer pthread ── renderSamples ──▶ planar float ring in SHARED wasm memory
 *   AudioWorkletProcessor (browser audio thread, plain JS) ── reads the ring ──▶ destination
 *
 * The ring lives in the module's own memory, which is a SharedArrayBuffer in a threaded build, so the
 * worklet reads the mixer's output in place: no copy between threads, no message per quantum. The mixer
 * keeps the ring filled to mBufferSize frames and sleeps on a futex; the worklet advances the read
 * position and wakes it with Atomics.notify after every 128-frame quantum.
 *
 * Requirements: -pthread (the ring must be shared memory), a cross-origin-isolated page (SAB), and
 * AudioWorklet. Anything else fails open() with NoDevice, so the next backend (null) is tried.
 *
 * Every JS call that touches the AudioContext runs on the browser main thread (OnMainThread), the
 * only thread Web Audio exists on. They happen at open/reset/start/stop, never per mix.
 *
 * CHANNELS. openal-soft renders at the channel count the output can actually play; the destination is
 * never fed a layout it will discard. Web Audio downmixes only 1/2/4/6 channels ("speakers"
 * interpretation, which is kept on both the node and the destination); 8 channels have no downmix rule,
 * so 7.1 is chosen only when the destination reports >= 8. Headphones cannot be detected from a page, so
 * DirectEar is never set and HRTF stays off unless the application or the config asks for it.
 */

#include "config.h"

#include "web.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <exception>
#include <thread>
#include <vector>

#include <emscripten/emscripten.h>
#include <emscripten/proxying.h>
#include <emscripten/threading.h>
#include <functional>

#include "althrd_setname.h"
#include "core/device.h"
#include "core/helpers.h"
#include "core/logging.h"


namespace {

using namespace std::string_view_literals;

[[nodiscard]] constexpr auto GetDeviceName() noexcept { return "Web Audio"sv; }

/* Shared control block, int32 words. The worklet sees the same memory. */
enum Ctrl : unsigned {
    ReadPos = 0,        /* frames consumed (mod 2^32) - worklet writes */
    WritePos,           /* frames produced (mod 2^32) - mixer writes */
    UnderrunEvents,     /* worklet: times the ring ran dry while playing */
    UnderrunFrames,     /* worklet: frames of silence those cost */
    Quanta,             /* worklet: render quanta served */
    Primed,             /* worklet: 1 while playing, 0 while (re)filling to the prime level */
    Stopped,            /* backend: 1 = the worklet must output nothing and end */
    MixerChunks,        /* mixer: chunks rendered */
    MixerWorstMicros,   /* mixer: slowest chunk, microseconds */
    Generation,         /* backend: bumped per start(); a processor from an earlier start() ends itself */
    MixerWorstGapMicros,/* mixer: longest time between two renders while the ring was below target */
    WorkletMaxBurst,    /* worklet: most frames pulled within 2 ms of wall time (the device's callback size) */
    NumCtrl
};

/* The AudioWorkletProcessor. Loaded from a Blob URL, so it is plain text here. */
constexpr const char WorkletSource[] = R"JS(
class AlsoftWebProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const p = options.processorOptions;
    this.i32 = new Int32Array(p.sab);
    this.f32 = new Float32Array(p.sab);
    this.c = p.ctrl >>> 2;
    this.d = p.data >>> 2;
    this.cap = p.cap;
    this.ch = p.ch;
    this.prime = p.prime;
    this.gen = p.gen;
    this.primed = false;
    this.burstStart = 0;
    this.burst = 0;
  }
  process(inputs, outputs) {
    const i32 = this.i32, c = this.c, out = outputs[0];
    if (i32[c + 6] || i32[c + 9] !== this.gen) return false;
    const n = out[0].length;
    const now = Date.now();
    if (now - this.burstStart > 2) { this.burstStart = now; this.burst = 0; }
    this.burst += n;
    if (this.burst > i32[c + 11]) i32[c + 11] = this.burst;
    const r = i32[c] >>> 0, w = i32[c + 1] >>> 0;
    const avail = (w - r) >>> 0;
    if (!this.primed) {
      if (avail < this.prime) return true;
      this.primed = true;
      i32[c + 5] = 1;
    }
    let take = n;
    if (avail < n) {
      /* Ran dry: play what there is, then refill to the prime level before playing again, so one
       * starved mixer costs one gap rather than a burst of 3 ms fragments. */
      take = avail;
      Atomics.add(i32, c + 2, 1);
      Atomics.add(i32, c + 3, n - take);
      this.port.postMessage({ t: currentTime, avail: avail });
      this.primed = false;
      i32[c + 5] = 0;
    }
    const pos = r & (this.cap - 1);
    const first = Math.min(take, this.cap - pos);
    const chans = Math.min(out.length, this.ch);
    for (let ch = 0; ch < chans; ch++) {
      const base = this.d + ch * this.cap, o = out[ch];
      if (first > 0) o.set(this.f32.subarray(base + pos, base + pos + first));
      if (first < take) o.set(this.f32.subarray(base, base + take - first), first);
      if (take < n) o.fill(0, take);
    }
    Atomics.store(i32, c, (r + take) | 0);
    Atomics.add(i32, c + 4, 1);
    Atomics.notify(i32, c);
    return true;
  }
}
registerProcessor("alsoft-web", AlsoftWebProcessor);
)JS";


/* The JavaScript half. EM_JS rather than EM_ASM: EM_ASM splits its code argument at every comma inside
 * braces (the preprocessor only respects parentheses), so an object literal breaks it. Each of these runs
 * on the browser main thread, reached through OnMainThread() below. */
EM_JS_DEPS(alsoft_web, "$UTF8ToString");

EM_JS(int, alsoft_web_open, (const char *source), {
    if (typeof AudioContext === "undefined" || typeof AudioWorkletNode === "undefined"
        || typeof SharedArrayBuffer === "undefined" || !(wasmMemory.buffer instanceof SharedArrayBuffer))
        return 0;
    const S = globalThis.__alsoftWeb || (globalThis.__alsoftWeb = {});
    S.worklet = UTF8ToString(source);
    return 1;
});

/* Returns (sampleRate << 6) | destination.maxChannelCount. */
EM_JS(int, alsoft_web_reset, (int want), {
    const S = globalThis.__alsoftWeb;
    if (S.ctx && want && S.ctx.sampleRate !== want) {
        S.ctx.close();
        S.ctx = null;
    }
    if (!S.ctx) {
        try {
            S.ctx = want ? new AudioContext({ sampleRate: want, latencyHint: "interactive" })
                : new AudioContext({ latencyHint: "interactive" });
        } catch (e) {
            S.ctx = new AudioContext({ latencyHint: "interactive" });
        }
        S.ready = S.ctx.audioWorklet.addModule(
            URL.createObjectURL(new Blob([S.worklet], { type: "application/javascript" })));
        /* Autoplay: a context created without a recent gesture starts suspended. Resume on the next
         * one; a page that already had one (the launch click) is usually running already. */
        const resume = () => { if (S.ctx && S.ctx.state !== "running") S.ctx.resume(); };
        for (const ev of ["pointerdown", "keydown", "touchend"])
            document.addEventListener(ev, resume, { capture: true, passive: true });
        resume();
    }
    return (S.ctx.sampleRate << 6) | Math.min(S.ctx.destination.maxChannelCount, 63);
});

EM_JS(void, alsoft_web_start, (int ctrl, int data, int cap, int ch, int prime, int gen), {
    const S = globalThis.__alsoftWeb;
    const opts = { sab: wasmMemory.buffer, ctrl: ctrl, data: data, cap: cap, ch: ch, prime: prime, gen: gen };
    const i32 = new Int32Array(wasmMemory.buffer);
    const c = ctrl >>> 2;
    S.stats = () => ({
        library: "openal-soft-web",
        sampleRate: S.ctx ? S.ctx.sampleRate : 0,
        channels: ch,
        destinationChannels: S.ctx ? S.ctx.destination.channelCount : 0,
        maxChannelCount: S.ctx ? S.ctx.destination.maxChannelCount : 0,
        ctxState: S.ctx ? S.ctx.state : "none",
        baseLatency: S.ctx ? S.ctx.baseLatency : 0,
        outputLatency: S.ctx && S.ctx.outputLatency !== undefined ? S.ctx.outputLatency : null,
        underrunEvents: i32[c + 2],
        underrunFrames: i32[c + 3],
        quanta: i32[c + 4],
        playing: i32[c + 5] === 1,
        mixerChunks: i32[c + 7],
        mixerWorstMs: i32[c + 8] / 1000,
        mixerWorstGapMs: i32[c + 10] / 1000,
        workletMaxBurst: i32[c + 11],
        ringFrames: ((i32[c + 1] >>> 0) - (i32[c] >>> 0)) >>> 0,
        bufferFrames: prime,
    });
    /* Test hook: empty the ring as the worklet sees it, so a harness can prove the underrun counter
     * counts (the positive control for "0 underruns"). Costs one glitch; nothing calls it in play. */
    S.drainForTest = () => { Atomics.store(i32, c + 1, Atomics.load(i32, c)); };
    S.ready.then(() => {
        if (i32[c + 6] || i32[c + 9] !== gen) return;
        const node = new AudioWorkletNode(S.ctx, "alsoft-web", {
            numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [ch], processorOptions: opts });
        node.channelInterpretation = "speakers";
        const dest = S.ctx.destination;
        if (ch > 2 && dest.maxChannelCount >= ch) dest.channelCount = ch;
        dest.channelInterpretation = "speakers";
        node.connect(dest);
        S.underrunLog = [];
        node.port.onmessage = (e) => { if (S.underrunLog.length < 500) S.underrunLog.push(e.data); };
        S.node = node;
    });
});

EM_JS(void, alsoft_web_stop, (), {
    const S = globalThis.__alsoftWeb;
    if (S && S.node) {
        S.node.disconnect();
        S.node = null;
    }
});

/* Run f on the browser main thread and wait for it. Web Audio exists only there. */
void OnMainThread(std::function<void()> f)
{
    if(emscripten_is_main_runtime_thread())
        return f();
    emscripten_proxy_sync(emscripten_proxy_get_system_queue(), emscripten_main_runtime_thread_id(),
        [](void *arg) { (*static_cast<std::function<void()>*>(arg))(); }, &f);
}


struct WebBackend final : BackendBase {
    explicit WebBackend(gsl::not_null<DeviceBase*> const device) noexcept : BackendBase{device}
    { }
    ~WebBackend() override;

    void mixerProc();

    void open(std::string_view name) override;
    auto reset() -> bool override;
    void start() override;
    void stop() override;

    std::atomic<bool> mKillNow{true};
    std::thread mThread;

    std::array<std::int32_t,NumCtrl> mCtrl{};
    std::vector<float> mRing;
    unsigned mCapacity{0u};
    unsigned mChannels{0u};
};

WebBackend::~WebBackend()
{ stop(); }

void WebBackend::mixerProc()
{
    SetRTPriority();
    althrd_setname(GetMixerThreadName());

    const auto update = mDevice->mUpdateSize;
    const auto target = mDevice->mBufferSize;
    const auto cap = mCapacity;
    const auto nch = mChannels;
    auto ptrs = std::array<void*,MaxOutputChannels>{};
    auto lastRender = emscripten_get_now();

    while(!mKillNow.load(std::memory_order_acquire)
        && mDevice->Connected.load(std::memory_order_acquire))
    {
        const auto r = static_cast<std::uint32_t>(mCtrl[ReadPos]);
        const auto w = static_cast<std::uint32_t>(mCtrl[WritePos]);
        if(w - r >= target)
        {
            /* Full enough. Sleep until the worklet consumes a quantum (it notifies ReadPos). The
             * timeout only bounds how long a stop request can go unnoticed. */
            emscripten_futex_wait(&mCtrl[ReadPos], static_cast<std::uint32_t>(r), 50.0);
            continue;
        }

        const auto t0 = emscripten_get_now();
        const auto gap = static_cast<std::int32_t>((t0-lastRender) * 1000.0);
        if(gap > mCtrl[MixerWorstGapMicros])
            mCtrl[MixerWorstGapMicros] = gap;
        const auto pos = w & (cap-1u);
        const auto first = std::min(update, cap-pos);
        for(auto c = 0u;c < nch;++c)
            ptrs[c] = &mRing[c*cap + pos];
        mDevice->renderSamples(std::span{ptrs.data(), nch}, first);
        if(first < update)
        {
            for(auto c = 0u;c < nch;++c)
                ptrs[c] = &mRing[c*cap];
            mDevice->renderSamples(std::span{ptrs.data(), nch}, update-first);
        }
        mCtrl[WritePos] = static_cast<std::int32_t>(w + update);

        lastRender = emscripten_get_now();
        const auto micros = static_cast<std::int32_t>((lastRender-t0) * 1000.0);
        mCtrl[MixerChunks] += 1;
        if(micros > mCtrl[MixerWorstMicros])
            mCtrl[MixerWorstMicros] = micros;
    }
}


void WebBackend::open(std::string_view name)
{
    if(name.empty())
        name = GetDeviceName();
    else if(name != GetDeviceName())
        throw al::backend_exception{al::backend_error::NoDevice, "Device name \"{}\" not found",
            name};

    auto ok = 0;
    OnMainThread([&ok] { ok = alsoft_web_open(WorkletSource); });
    if(!ok)
        throw al::backend_exception{al::backend_error::NoDevice,
            "Web Audio output needs AudioWorklet and shared wasm memory"};

    mDeviceName = name;
}

auto WebBackend::reset() -> bool
{
    /* The AudioContext is created (or re-created) here rather than in open(), because only now is the
     * rate the application asked for known. Honouring it lets the browser do the one resample to the
     * hardware rate natively instead of openal-soft mixing at a rate the app did not choose. */
    const auto wantRate = mDevice->mFlags.test(DeviceFlag::FrequencyRequest) ? mDevice->mSampleRate : 0u;
    auto packed = 0;
    OnMainThread([&packed,wantRate] { packed = alsoft_web_reset(static_cast<int>(wantRate)); });
    const auto rate = packed >> 6;
    const auto maxChannels = static_cast<unsigned>(packed & 63);

    if(rate < static_cast<int>(MinOutputRate) || rate > static_cast<int>(MaxOutputRate))
        throw al::backend_exception{al::backend_error::DeviceError, "Unhandled sample rate: {}",
            rate};
    mDevice->mSampleRate = static_cast<unsigned>(rate);

    /* Channel layout: what was asked for if the output can play it, else the largest layout it can
     * play that Web Audio knows how to downmix, stereo at the bottom. */
    auto chans = mDevice->FmtChans;
    const auto fits = [maxChannels](DevFmtChannels f)
    { return ChannelsFromDevFmt(f, 0) <= maxChannels; };
    if(!mDevice->mFlags.test(DeviceFlag::ChannelsRequest) || chans == DevFmtAmbi3D || chans == DevFmtX61
        || chans == DevFmtX714 || chans == DevFmtX7144 || chans == DevFmtX3D71 || !fits(chans))
    {
        if(maxChannels >= 8) chans = DevFmtX71;
        else if(maxChannels >= 6) chans = DevFmtX51;
        else if(maxChannels >= 4 && mDevice->mFlags.test(DeviceFlag::ChannelsRequest)) chans = DevFmtQuad;
        else if(maxChannels >= 2) chans = DevFmtStereo;
        else chans = DevFmtMono;
    }
    mDevice->FmtChans = chans;
    mDevice->mAmbiOrder = 0;
    mDevice->FmtType = DevFmtFloat;

    /* 10 ms chunks, 30 ms ring target: under the 40 ms added-latency budget, and deep enough to ride
     * through a cold-JIT chunk (measured 8-20 ms in the first second, 2.61 §0.2). */
    mDevice->mUpdateSize = std::max(mDevice->mSampleRate / 100u, 128u);
    mDevice->mBufferSize = mDevice->mUpdateSize * 3u;

    setDefaultWFXChannelOrder();
    TRACE("Web Audio: {}hz, {} channel(s) (destination max {}), update {}, buffer {}",
        mDevice->mSampleRate, mDevice->channelsFromFmt(), maxChannels, mDevice->mUpdateSize,
        mDevice->mBufferSize);
    return true;
}

void WebBackend::start()
{
    mChannels = mDevice->channelsFromFmt();
    mCapacity = std::bit_ceil(mDevice->mBufferSize * 2u);
    mRing.assign(std::size_t{mCapacity} * mChannels, 0.0f);
    /* Everything but the generation restarts at zero. A processor left over from the previous start()
     * may still run a quantum or two after its node was disconnected; the new generation makes it end
     * itself instead of draining this ring alongside the new one. */
    const auto generation = mCtrl[Generation] + 1;
    mCtrl.fill(0);
    mCtrl[Generation] = generation;

    const auto ctrl = static_cast<int>(reinterpret_cast<std::uintptr_t>(mCtrl.data()));
    const auto data = static_cast<int>(reinterpret_cast<std::uintptr_t>(mRing.data()));
    OnMainThread([=,this] {
        alsoft_web_start(ctrl, data, static_cast<int>(mCapacity), static_cast<int>(mChannels),
            static_cast<int>(mDevice->mBufferSize), generation);
    });

    try {
        mKillNow.store(false, std::memory_order_release);
        mThread = std::thread{&WebBackend::mixerProc, this};
    }
    catch(std::exception& e) {
        throw al::backend_exception{al::backend_error::DeviceError,
            "Failed to start mixing thread: {}", e.what()};
    }
}

void WebBackend::stop()
{
    if(mKillNow.exchange(true, std::memory_order_acq_rel) || !mThread.joinable())
        return;
    mCtrl[Stopped] = 1;
    emscripten_futex_wake(&mCtrl[ReadPos], 1);
    mThread.join();

    OnMainThread([] { alsoft_web_stop(); });
}

} // namespace


auto WebBackendFactory::init() -> bool
{ return true; }

auto WebBackendFactory::querySupport(BackendType const type) -> bool
{ return (type == BackendType::Playback); }

auto WebBackendFactory::enumerate(BackendType const type) -> std::vector<std::string>
{
    switch(type)
    {
    case BackendType::Playback:
        return std::vector{std::string{GetDeviceName()}};
    case BackendType::Capture:
        break;
    }
    return {};
}

auto WebBackendFactory::createBackend(gsl::not_null<DeviceBase*> const device,
    BackendType const type) -> BackendPtr
{
    if(type == BackendType::Playback)
        return BackendPtr{new WebBackend{device}};
    return nullptr;
}

auto WebBackendFactory::getFactory() -> BackendFactory&
{
    static WebBackendFactory factory{};
    return factory;
}
