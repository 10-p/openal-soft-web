/**
 * ufront (10-p/openal-soft-web): AL_UFRONT_galaxy_reverb — the property handler for the Galaxy Sound
 * System sample reverb. The DSP is in alc/effects/galaxyreverb.cpp; the parameters are ZoneInfo's as
 * UnGalaxy.cpp mapped them (see include/AL/alext.h).
 */

#include "config.h"

#include <algorithm>
#include <span>

#include "AL/al.h"
#include "AL/alext.h"
#include "AL/efx.h"

#include "alnumeric.h"
#include "altypes.hpp" // the _uz literal: under HAVE_CXXMODULES (Android) `import alc.context;` does not carry it
#include "effects.h"

#if HAVE_CXXMODULES
import alc.context;
#else
#include "alc/context.hpp"
#endif


namespace {

consteval auto genDefaultProps() noexcept -> EffectProps
{
    /* ZoneInfo defaultproperties: MasterGain=100, CutoffHz=6000, Delay(0)=20, Delay(1)=34, Gain(0)=150,
     * Gain(1)=70, the other four taps 0 — which UnGalaxy's clamps turn into 0.001 s / 0.001. */
    return GalaxyReverbProps{
        .Volume     = AL_GALAXY_REVERB_DEFAULT_VOLUME,
        .HFDamp     = AL_GALAXY_REVERB_DEFAULT_HFDAMP,
        .DelayTimes = {20.0f/500.0f, 34.0f/500.0f, 0.001f, 0.001f, 0.001f, 0.001f},
        .DelayGains = {150.0f/255.0f, 70.0f/255.0f, 0.001f, 0.001f, 0.001f, 0.001f}};
}

} // namespace

constinit const EffectProps GalaxyReverbEffectProps(genDefaultProps());

void GalaxyReverbEffectHandler::SetParami(al::Context *context, GalaxyReverbProps&, ALenum param, int)
{ context->throw_error(AL_INVALID_ENUM, "Invalid galaxy reverb integer property {:#04x}", as_unsigned(param)); }
void GalaxyReverbEffectHandler::SetParamiv(al::Context *context, GalaxyReverbProps&, ALenum param, const int*)
{ context->throw_error(AL_INVALID_ENUM, "Invalid galaxy reverb integer-vector property {:#04x}", as_unsigned(param)); }
void GalaxyReverbEffectHandler::SetParamf(al::Context *context, GalaxyReverbProps &props, ALenum param, float val)
{
    switch(param)
    {
    case AL_GALAXY_REVERB_VOLUME:
        if(!(val >= AL_GALAXY_REVERB_MIN_VOLUME && val <= AL_GALAXY_REVERB_MAX_VOLUME))
            context->throw_error(AL_INVALID_VALUE, "Galaxy reverb volume out of range");
        props.Volume = val;
        return;

    case AL_GALAXY_REVERB_HFDAMP:
        if(!(val >= AL_GALAXY_REVERB_MIN_HFDAMP && val <= AL_GALAXY_REVERB_MAX_HFDAMP))
            context->throw_error(AL_INVALID_VALUE, "Galaxy reverb HF damping out of range");
        props.HFDamp = val;
        return;
    }

    context->throw_error(AL_INVALID_ENUM, "Invalid galaxy reverb float property {:#04x}",
        as_unsigned(param));
}
void GalaxyReverbEffectHandler::SetParamfv(al::Context *context, GalaxyReverbProps &props, ALenum param, const float *vals)
{
    switch(param)
    {
    case AL_GALAXY_REVERB_DELAY_TIMES:
    {
        auto const in = std::span{vals, 6_uz};
        if(!std::ranges::all_of(in, [](float const v){ return v >= AL_GALAXY_REVERB_MIN_DELAY_TIME && v <= AL_GALAXY_REVERB_MAX_DELAY_TIME; }))
            context->throw_error(AL_INVALID_VALUE, "Galaxy reverb delay time out of range");
        std::ranges::copy(in, props.DelayTimes.begin());
        return;
    }
    case AL_GALAXY_REVERB_DELAY_GAINS:
    {
        auto const in = std::span{vals, 6_uz};
        if(!std::ranges::all_of(in, [](float const v){ return v >= AL_GALAXY_REVERB_MIN_DELAY_GAIN && v <= AL_GALAXY_REVERB_MAX_DELAY_GAIN; }))
            context->throw_error(AL_INVALID_VALUE, "Galaxy reverb delay gain out of range");
        std::ranges::copy(in, props.DelayGains.begin());
        return;
    }
    }
    SetParamf(context, props, param, *vals);
}

void GalaxyReverbEffectHandler::GetParami(al::Context *context, const GalaxyReverbProps&, ALenum param, int*)
{ context->throw_error(AL_INVALID_ENUM, "Invalid galaxy reverb integer property {:#04x}", as_unsigned(param)); }
void GalaxyReverbEffectHandler::GetParamiv(al::Context *context, const GalaxyReverbProps&, ALenum param, int*)
{ context->throw_error(AL_INVALID_ENUM, "Invalid galaxy reverb integer-vector property {:#04x}", as_unsigned(param)); }
void GalaxyReverbEffectHandler::GetParamf(al::Context *context, const GalaxyReverbProps &props, ALenum param, float *val)
{
    switch(param)
    {
    case AL_GALAXY_REVERB_VOLUME: *val = props.Volume; return;
    case AL_GALAXY_REVERB_HFDAMP: *val = props.HFDamp; return;
    }

    context->throw_error(AL_INVALID_ENUM, "Invalid galaxy reverb float property {:#04x}",
        as_unsigned(param));
}
void GalaxyReverbEffectHandler::GetParamfv(al::Context *context, const GalaxyReverbProps &props, ALenum param, float *vals)
{
    switch(param)
    {
    case AL_GALAXY_REVERB_DELAY_TIMES: std::ranges::copy(props.DelayTimes, vals); return;
    case AL_GALAXY_REVERB_DELAY_GAINS: std::ranges::copy(props.DelayGains, vals); return;
    }
    GetParamf(context, props, param, vals);
}
