/**
 * ufront (10-p/openal-soft-web): the Galaxy Sound System's sample reverb as an openal-soft effect
 * (AL_UFRONT_galaxy_reverb, AL_EFFECT_GALAXY_REVERB_UFRONT).
 *
 * WHAT IT IS. The "classic" zone reverb of Unreal (1998) and Unreal Tournament (1999): three stereo
 * Schroeder all-pass stages in series, each with a per-channel tap delay (Delay[2k] for L, Delay[2k+1]
 * for R, read from one interleaved ring written once per frame) and a one-pole low-pass in its feedback
 * whose single state is threaded through the three stages in turn (stage k filters with the state stage
 * k-1 left; stage 1 with what stage 3 left the frame before). The stage-3 output of the previous frame is
 * fed back CROSS-CHANNEL into the input (halved), which closes a loop around the whole network — the
 * "hanging" tail Unreal players remember comes from there. Galaxy mixed the wet signal into its dry
 * output in place from a separate send buffer; here the slot input is the send and the dry path is
 * openal-soft's own.
 *
 * WHERE IT COMES FROM. ut99-v400/Galaxy/Debug/GALAXY.LIB (Epic's UT v400 source drop): `mmxReverb`
 * (mmx.obj, 465 bytes of MMX, read by hand — Q15 pmaddwd / psrad 15 / packssdw with paddsw saturation)
 * and `glxSetMMXReverb` (Galaxy.obj, x87 C, decompiled with Ghidra for the coefficient formulas).
 * `x86Reverb`, the plain-x86 routine, is an EMPTY STUB (push ebp; pusha; popa; leave; ret) in the 1998
 * and 1999 objects alike: Galaxy had no reverb at all on a CPU without MMX. The SSE (kniReverb) and
 * 3DNow! (k3dReverb) variants are a different, six-ring network and are not recovered.
 *
 * FIDELITY. The loop is 16-bit fixed point, so this C is bit-exact with the original object on every
 * target (verified under wine against mmx.obj: 0 mismatches over 177,725 frames on four parameter
 * sets, rings and state identical) once two call-boundary quirks of the original are emulated — the
 * first frame of every call reloads the saved output with `movd` (zero-extended, so the R channel's
 * cross-feed word is 0 for that frame) and fetches stage 1's taps at delay instead of delay+1. Those
 * are block-size artifacts, not the sound; this implementation keeps the loop's steady state on every
 * frame. The coefficient setup mirrors the original's x87 arithmetic in double: over a 4,000-case
 * parameter sweep against the original, 47,999 of 48,000 Q15 coefficients matched and one differed
 * by a single LSB (extended-precision rounding). ★ Galaxy's "two pi" is the float 6.282, not 2π —
 * measured, and the only way the sweep matched.
 *
 * INPUT. The slot's ambisonic wet mix is decoded to a virtual stereo pair, L = (W + Y/√3)/2 and
 * R = (W − Y/√3)/2 (N3D, ACN 1 = Y, +Y = left), so a source on the left drives the left taps first,
 * as Galaxy's panned stereo send did; the output pair is panned hard left / hard right.
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
constexpr auto GlxRing = 16384_uz;             /* frames per stage ring: 0x4000, masked with 0x3fff */
constexpr auto GlxStages = 3_uz;               /* three stereo all-pass stages; six taps = 3 x {L,R} */

struct GlxFrame16 { int16_t l, r; };

struct GlxState {
    int32_t widx;                              /* ring write index (shared by the 3 rings) */
    std::array<int32_t,6> delay;               /* NEGATIVE tap offsets in frames: -(int)(rate * Time[i]) */
    std::array<std::array<int16_t,4>,GlxStages> a;   /* per stage {1-gL^2, -gL, 1-gR^2, -gR}  Q15 */
    std::array<std::array<int16_t,4>,GlxStages> b;   /* per stage {gL*d, 1-d, gR*d, 1-d}      Q15 */
    std::array<int16_t,4> mix;                 /* {dryL=0x7fff, wetL, dryR=0x7fff, wetR}  Q15 */
    std::array<int16_t,2> z;                   /* the shared one-pole low-pass state (L,R) */
    std::array<int16_t,2> y;                   /* the previous frame's stage-3 output (L,R), cross-fed */
    std::array<std::array<GlxFrame16,GlxRing>,GlxStages> ring;
};

constexpr auto sat16(int32_t const v) noexcept -> int16_t
{ return static_cast<int16_t>(v > 32767 ? 32767 : (v < -32768 ? -32768 : v)); }
constexpr auto asr15(int32_t const v) noexcept -> int32_t { return v >> 15; }      /* psrad: arithmetic */
inline auto q15(double const v) noexcept -> int16_t { return static_cast<int16_t>(static_cast<int32_t>(v)); } /* MSVC __ftol: truncate */

/* glxSetMMXReverb: Volume -1..1, HFDamp <= rate, six {time s, gain} taps. Clears the rings, as Galaxy did. */
void glxSetup(GlxState &st, float const volume, float hfdamp, std::span<const float,6> const times,
    std::span<const float,6> const gains, unsigned const rate)
{
    std::memset(&st, 0, sizeof st);
    hfdamp = std::min(hfdamp, static_cast<float>(rate));
    auto const frate = static_cast<float>(rate);
    st.mix[0] = st.mix[2] = 0x7fff;
    st.mix[1] = st.mix[3] = q15(volume * 32767.0);
    /* one-pole low-pass: w = cutoff * 6.282 / rate (Galaxy's "two pi" is the float 6.282, measured, not 2*pi),
     * t = 1 - cos(w), d = sqrt((t+2)t) - t  (== 1 - (x - sqrt(x^2-1)), x = 2 - cos w) */
    auto const t = static_cast<float>(1.0 - std::cos(static_cast<double>(hfdamp * 6.282f / frate)));
    auto const d = static_cast<float>(std::sqrt(static_cast<double>((t + 2.0f) * t)) - t);
    for(auto i = 0_uz;i < 6;++i)
    {
        auto g = 1.0f - gains[i];
        auto T = times[i];
        if(T == 0.0f) T = 0.001f;
        if(g == 1.0f) g = 0.999f;
        auto frames = static_cast<int32_t>(frate * T);
        frames = std::min(frames, static_cast<int32_t>(GlxRing - 2));   /* a rate above 48 kHz would overrun the ring */
        st.delay[i] = -frames;
        auto const k = i >> 1, c = (i & 1) * 2;        /* L: words 0,1 — R: words 2,3 */
        st.b[k][c]     = q15(static_cast<double>(g) * d * 32767.0);
        st.b[k][c + 1] = q15((1.0 - static_cast<double>(d)) * 32767.0);
        st.a[k][c + 1] = q15(static_cast<double>(g) * -32767.0);
        st.a[k][c]     = q15((1.0 - static_cast<double>(g * g)) * 32767.0);
    }
}

/* mmxReverb: in place, dst (the dry mix) += reverb of src (the effect send), n stereo int16 frames. */
void glxProcess(GlxState &st, std::span<GlxFrame16> const dst, std::span<const GlxFrame16> const src)
{
    auto idx = st.widx;
    auto yL = st.y[0], yR = st.y[1], zL = st.z[0], zR = st.z[1];
    auto const n = std::min(dst.size(), src.size());
    for(auto f = 0_uz;f < n;++f)
    {
        /* input pre-mix: half of (send + the previous output, cross-channel) — paddsw then psraw 1 */
        auto cL = static_cast<int16_t>(sat16(static_cast<int32_t>(yR) + src[f].l) >> 1);
        auto cR = static_cast<int16_t>(sat16(static_cast<int32_t>(yL) + src[f].r) >> 1);
        for(auto k = 0_uz;k < GlxStages;++k)
        {
            auto &ring = st.ring[k];
            /* Stage 1's taps are one frame older than its delay says: the original prefetches the next
             * frame's stage-1 taps at the END of the loop body, before it advances the write index. */
            auto const skew = (k == 0) ? -1 : 0;
            auto const tL = ring[static_cast<size_t>(idx + st.delay[2*k] + skew) & (GlxRing - 1)].l;
            auto const tR = ring[static_cast<size_t>(idx + st.delay[2*k + 1] + skew) & (GlxRing - 1)].r;
            auto const nyL = sat16(asr15(tL * st.a[k][0] + cL * st.a[k][1]));
            auto const nyR = sat16(asr15(tR * st.a[k][2] + cR * st.a[k][3]));
            auto const nzL = sat16(asr15(tL * st.b[k][0] + zL * st.b[k][1]));
            auto const nzR = sat16(asr15(tR * st.b[k][2] + zR * st.b[k][3]));
            ring[static_cast<size_t>(idx)].l = sat16(static_cast<int32_t>(cL) + nzL);
            ring[static_cast<size_t>(idx)].r = sat16(static_cast<int32_t>(cR) + nzR);
            cL = nyL; cR = nyR; zL = nzL; zR = nzR;
        }
        dst[f].l = sat16(asr15(dst[f].l * st.mix[0] + cL * st.mix[1]));
        dst[f].r = sat16(asr15(dst[f].r * st.mix[2] + cR * st.mix[3]));
        yL = cL; yR = cR;
        idx = (idx + 1) & static_cast<int32_t>(GlxRing - 1);
    }
    st.widx = idx; st.y[0] = yL; st.y[1] = yR; st.z[0] = zL; st.z[1] = zR;
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

    alignas(16) std::array<FloatBufferLine,2> mTempBuffer{};
    std::array<GlxFrame16,BufferLineSize> mIn{};
    std::array<GlxFrame16,BufferLineSize> mOut{};

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
    mRate = device->mSampleRate;
    /* A parameter change re-runs the setup, which clears the rings — exactly what glxSetSampleReverb did
     * on every zone change. Unchanged parameters (the slot's own gain moved, or a re-attach) keep the tail. */
    if(std::memcmp(&props, &mProps, sizeof props) != 0)
    {
        mProps = props;
        glxSetup(mDsp, mProps.Volume, mProps.HFDamp, mProps.DelayTimes, mProps.DelayGains, mRate);
    }

    /* The wet pair leaves hard left / hard right, like Galaxy's stereo mix. */
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
    for(auto i = 0_uz;i < samplesToDo;++i)
    {
        auto const w = samplesIn[0][i] * 0.5f;
        auto const y = haveY ? samplesIn[1][i] * side : 0.0f;
        mIn[i].l = sat16(static_cast<int32_t>(std::lround((w + y) * 32767.0f)));
        mIn[i].r = sat16(static_cast<int32_t>(std::lround((w - y) * 32767.0f)));
        mOut[i].l = 0; mOut[i].r = 0;             /* no dry here: the mix quad's wet term alone comes out */
    }
    glxProcess(mDsp, std::span{mOut}.first(samplesToDo), std::span{mIn}.first(samplesToDo));
    for(auto i = 0_uz;i < samplesToDo;++i)
    {
        mTempBuffer[0][i] = static_cast<float>(mOut[i].l) * (1.0f/32768.0f);
        mTempBuffer[1][i] = static_cast<float>(mOut[i].r) * (1.0f/32768.0f);
    }

    for(auto c = 0_uz;c < 2;++c)
        MixSamples(std::span{mTempBuffer[c]}.first(samplesToDo), samplesOut, mGains[c].Current,
            mGains[c].Target, samplesToDo, 0);
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
