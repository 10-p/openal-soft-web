/**
 * ufront (10-p/openal-soft-web): the Galaxy Sound System's sample reverb as an openal-soft effect
 * (AL_UFRONT_galaxy_reverb, AL_EFFECT_GALAXY_REVERB_UFRONT).
 *
 * WHAT IT IS. The "classic" zone reverb of Unreal (1998): six taps on one mono ring of 32,768 16-bit
 * cells, each weighted by its gain and summed in Q12, feed a one-pole low-pass; the low-pass output is the
 * wet signal, added to both channels, and it is also fed back into the ring together with the send
 * (L + R), scaled by the zone's MasterGain. A multi-tap echo with a damped feedback loop: the loop gain is
 * roughly Volume x sum(Gain), and every reverb zone Unreal 1 and UT99 ship stays below 0.8.
 *
 * WHERE IT COMES FROM. unreal1-v200/Galaxy/Lib/Galaxy-s.lib (Epic's Unreal 1 v200 source drop):
 * `glxSetSampleReverb` calls `glxSetEffect` (Galaxy.obj, x87 C: the coefficients) and the software mixer
 * runs `mmxEffect` (mmx.obj, 301 bytes of MMX) on the sample send. That library's `mmxReverb` is its
 * MUSIC reverb (glxSetMusicReverb), not this one.
 *
 * FIDELITY. The loop and the setup are bit-exact with the original objects (differential test under wine,
 * ufrontend packages/e2e/tools/galaxy-reverb: 0 mismatches over 1,994,184 frames on twelve parameter sets,
 * ring and state identical; every state word equal over a 4,000-case parameter sweep at 11, 22, 44 and
 * 48 kHz). mmxEffect has no call-boundary quirks. Galaxy wrote `sat16(send + wet)` into its output; here
 * the wet part alone leaves the effect and openal-soft's own dry path carries the sound, so only that
 * final 16-bit clip of the sum is not reproduced.
 *
 * WHY NOT THE 1999 ROUTINE (ufront 2.70 shipped it, 2.73 replaced it). UT v400's GALAXY.LIB carries a
 * different sample reverb: three stereo Q15 all-pass stages (`mmxReverb`) and the same network in float
 * (`kniReverb` for SSE, `k3dReverb` for 3DNow!, identical instruction for instruction). Its MMX version has
 * a copy-paste defect: stages 2 and 3 reuse stage 1's low-pass register (MM4) while MM5 and MM6 are loaded
 * and saved but never used, and that shared state makes the loop run away into a loud, endless tone. The
 * original object does it too, on 44 of Unreal 1's 83 reverb zones and on UT99's CTF-November (NyLeve's
 * start among them). This routine was chosen over the correct SSE one by measurement: on a Cortex-A55
 * running armv7 code it costs 56 ns per frame against 81 (SSE float) and 141 (the 1999 MMX loop).
 *
 * INPUT. The slot's ambisonic wet mix is decoded to a virtual stereo pair, L = (W + Y/√3)/2 and
 * R = (W − Y/√3)/2 (N3D, ACN 1 = Y, +Y = left), converted to 16 bits as Galaxy's mixer produced them; the
 * mono wet signal leaves panned hard left and hard right at once, as Galaxy added it to both channels.
 */

#include "config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <variant>

#include "AL/alext.h"

#include "alc/effects/base.h"
#include "alnumeric.h"
#include "core/ambidefs.h"
#include "core/bufferline.h"
#include "core/context.h"
#include "core/device.h"
#include "core/effects/base.h"
#include "core/effectslot.h"
#include "core/mixer.h"
#include "intrusive_ptr.h"
#include "opthelpers.h"

struct BufferStorage;

namespace {

/* ---------------------------------------------------------------------------------------------------
 * The recovered DSP, as portable fixed-point C. Kept free of openal-soft types so it can be tested
 * on its own (ufrontend packages/e2e/tools/galaxy-reverb/ carries the same code with the harness).
 * ------------------------------------------------------------------------------------------------- */
constexpr auto GlxRing = 32768_uz;             /* 0x8000 int16 cells, masked with 0x7fff */

struct GlxFrame16 { int16_t l, r; };

/* The original's state block (0x10034 bytes), field for field. */
struct GlxState {
    int32_t widx;                              /* 0x00 ring write index */
    std::array<int16_t,2> m;                   /* 0x04 the two words of ftol(Volume*32767): ring input weights */
    std::array<int32_t,6> delay;               /* 0x08 NEGATIVE tap offsets in frames: -trunc(rate*Time[i]) */
    std::array<int16_t,6> w;                   /* 0x20 tap weights, Q12: trunc(Gain[i]*4095) */
    int16_t z;                                 /* 0x2c the one-pole low-pass state */
    int16_t zcopy;                             /* 0x2e the original stores z's low word twice; never read */
    std::array<int16_t,2> c;                   /* 0x30 {(1-d)*32767, d*32767} */
    std::array<int16_t,GlxRing> ring;          /* 0x34 */
};

constexpr auto sat16(int32_t const v) noexcept -> int16_t
{ return static_cast<int16_t>(v > 32767 ? 32767 : (v < -32768 ? -32768 : v)); }
constexpr auto ftol(double const v) noexcept -> int32_t { return static_cast<int32_t>(v); } /* MSVC __ftol: truncate */

/* glxSetEffect: Volume -1..1, HFDamp <= rate, six {time s, gain} taps. Clears the ring, as Galaxy did.
 * The x87 products are exact in double (a float times an integer), so double reproduces them. */
void glxSetup(GlxState &st, float const volume, float hfdamp, std::span<const float,6> const times,
    std::span<const float,6> const gains, unsigned const rate)
{
    std::memset(&st, 0, sizeof st);        /* not `st = GlxState{}`: a 64 KB temporary on the mixer thread's stack */
    hfdamp = std::min(hfdamp, static_cast<float>(rate));
    auto const mv = static_cast<uint32_t>(ftol(static_cast<double>(volume) * 32767.0));
    st.m[0] = static_cast<int16_t>(mv & 0xffff);
    st.m[1] = static_cast<int16_t>(mv >> 16);
    /* one-pole low-pass: w = cutoff * 6.282 / rate (Galaxy's "two pi" is the float 6.282, in 1998 as in 1999),
     * t = 1 - cos(w), d = sqrt((t+2)t) - t */
    auto const t = 1.0 - std::cos(static_cast<double>(hfdamp) * static_cast<double>(6.282f)
        / static_cast<double>(rate));
    auto const d = static_cast<float>(std::sqrt((t + 2.0) * t) - t);
    st.c[1] = static_cast<int16_t>(ftol(static_cast<double>(d) * 32767.0));
    st.c[0] = static_cast<int16_t>(ftol((1.0 - static_cast<double>(d)) * 32767.0));
    for(auto i = 0_uz;i < 6;++i)
    {
        auto frames = ftol(static_cast<double>(rate) * static_cast<double>(times[i]));
        frames = std::min(frames, static_cast<int32_t>(GlxRing - 1));   /* a rate above 64 kHz would wrap the ring */
        st.delay[i] = -frames;
        st.w[i] = static_cast<int16_t>(ftol(static_cast<double>(gains[i]) * 4095.0));
    }
}

/* mmxEffect: wet[f] = the low-pass output for each stereo int16 send frame; the ring takes the send back. */
void glxProcess(GlxState &st, std::span<int16_t> const wet, std::span<const GlxFrame16> const src)
{
    auto idx = st.widx;
    auto z = st.z;
    auto const n = std::min(wet.size(), src.size());
    for(auto f = 0_uz;f < n;++f)
    {
        /* six taps, PMADDWD + PADDD: 32-bit sum, then psrad 12 and packssdw */
        auto sum = int32_t{0};
        for(auto i = 0_uz;i < 6;++i)
            sum += int32_t{st.ring[static_cast<size_t>(idx + st.delay[i]) & (GlxRing - 1)]} * st.w[i];
        auto const s = sat16(sum >> 12);
        /* the low-pass keeps the low word of (z*c0 + s*c1) >> 15 — no saturation, as the original */
        z = static_cast<int16_t>((int32_t{z} * st.c[0] + int32_t{s} * st.c[1]) >> 15);
        wet[f] = z;
        /* ring input: paddsw (z+L), paddsw R; and (z+R) — weighted by the two words of the volume */
        auto const a = sat16(int32_t{sat16(int32_t{z} + src[f].l)} + src[f].r);
        auto const b = sat16(int32_t{z} + src[f].r);
        st.ring[static_cast<size_t>(idx)] = static_cast<int16_t>((int32_t{a} * st.m[0] + int32_t{b} * st.m[1]) >> 15);
        idx = (idx + 1) & static_cast<int32_t>(GlxRing - 1);
    }
    st.widx = idx;
    st.z = z;
}

/* ---------------------------------------------------------------------------------------------------
 * The effect around it.
 * ------------------------------------------------------------------------------------------------- */
struct GalaxyReverbState final : public EffectState {
    GlxState mDsp{};
    GalaxyReverbProps mProps{};
    unsigned mRate{44100};

    struct OutGains {
        std::array<float,MaxAmbiChannels> Current{};
        std::array<float,MaxAmbiChannels> Target{};
    };
    std::array<OutGains,2> mGains;

    alignas(16) FloatBufferLine mTempBuffer{};
    std::array<GlxFrame16,BufferLineSize> mIn{};
    std::array<int16_t,BufferLineSize> mWet{};

    void deviceUpdate(const DeviceBase *device, const BufferStorage *buffer) override;
    void update(const ContextBase *context, const EffectSlotBase *slot, const EffectProps *props,
        const EffectTarget target) override;
    void process(const size_t samplesToDo, const std::span<const FloatBufferLine> samplesIn,
        const std::span<FloatBufferLine> samplesOut) override;
};

void GalaxyReverbState::deviceUpdate(const DeviceBase *device, const BufferStorage*)
{
    mRate = device->mSampleRate;
    glxSetup(mDsp, mProps.Volume, mProps.HFDamp, mProps.DelayTimes, mProps.DelayGains, mRate);
    mGains.fill(OutGains{});
}

void GalaxyReverbState::update(const ContextBase *context, const EffectSlotBase *slot,
    const EffectProps *props_, const EffectTarget target)
{
    auto const &props = std::get<GalaxyReverbProps>(*props_);
    auto const device = al::get_not_null(context->mDevice);
    /* A parameter change re-runs the setup, which clears the ring — exactly what glxSetEffect did on every
     * zone change. Unchanged parameters (the slot's own gain moved, or a re-attach) keep the tail. */
    if(device->mSampleRate != mRate || props.Volume != mProps.Volume || props.HFDamp != mProps.HFDamp
        || props.DelayTimes != mProps.DelayTimes || props.DelayGains != mProps.DelayGains)
    {
        mRate = device->mSampleRate;
        mProps = props;
        glxSetup(mDsp, mProps.Volume, mProps.HFDamp, mProps.DelayTimes, mProps.DelayGains, mRate);
    }

    /* The wet signal leaves hard left and hard right, as Galaxy added it to both channels. */
    auto const coeffs0 = CalcAmbiCoeffs( 1.0f, 0.0f, 0.0f, 0.0f);
    auto const coeffs1 = CalcAmbiCoeffs(-1.0f, 0.0f, 0.0f, 0.0f);

    mOutTarget = target.Main->Buffer;
    ComputePanGains(target.Main, coeffs0, slot->Gain, mGains[0].Target);
    ComputePanGains(target.Main, coeffs1, slot->Gain, mGains[1].Target);
}

void GalaxyReverbState::process(const size_t samplesToDo, const std::span<const FloatBufferLine> samplesIn,
    const std::span<FloatBufferLine> samplesOut)
{
    ASSUME(samplesToDo > 0);
    constexpr auto side = 0.5f / 1.7320508f;     /* Y is N3D-scaled: /sqrt(3) puts a hard-side source at +-1 */
    auto const haveY = samplesIn.size() > 1;
    auto const to16 = [](float const v) noexcept -> int16_t
    { return static_cast<int16_t>(fastf2i(std::clamp(v * 32767.0f, -32768.0f, 32767.0f))); };
    for(auto i = 0_uz;i < samplesToDo;++i)
    {
        auto const w = samplesIn[0][i] * 0.5f;
        auto const y = haveY ? samplesIn[1][i] * side : 0.0f;
        mIn[i].l = to16(w + y);
        mIn[i].r = to16(w - y);
    }
    glxProcess(mDsp, std::span{mWet}.first(samplesToDo), std::span{mIn}.first(samplesToDo));
    for(auto i = 0_uz;i < samplesToDo;++i)
        mTempBuffer[i] = static_cast<float>(mWet[i]) * (1.0f/32768.0f);

    auto const wet = std::span{mTempBuffer}.first(samplesToDo);
    for(auto c = 0_uz;c < 2;++c)
        MixSamples(wet, samplesOut, mGains[c].Current, mGains[c].Target, samplesToDo, 0);
}


struct GalaxyReverbStateFactory final : public EffectStateFactory {
    al::intrusive_ptr<EffectState> create() override
    { return al::intrusive_ptr<EffectState>{new GalaxyReverbState{}}; }
};

} // namespace

auto GalaxyReverbStateFactory_getFactory() -> gsl::not_null<EffectStateFactory*>
{
    static GalaxyReverbStateFactory GalaxyReverbFactory{};
    return gsl::make_not_null(&GalaxyReverbFactory);
}
