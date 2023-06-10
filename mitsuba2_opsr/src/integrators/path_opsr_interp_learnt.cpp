#include <enoki/stl.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/roughening/opsr_path.h>
#include <mitsuba/render/roughening/opsr_data.h>
#include <random>

NAMESPACE_BEGIN(mitsuba)

/**
 *  IMPORTANT : This integrator is meant to be used directly for rendering and not for the optimisation process.
 *  We offer two set of pre-optimised attenuation factors (for different bias-variance trade-offs) that can be 
 *  set using the 'roughening_mode' option:
 *      - 'opsr_low' beta = 0.001, spp = 1024 (for the optimisation), learning rate=0.0005 
 *      - 'opsr_moderate' beta = 0.05, spp = 1024 (for the optimisation), learning rate=0.0005 
 * 
 *  If you want to optimise the parameters yourself use 'path_opsr_interp' instead.
 * 
 *  This integrator implements Optimised Path Space Regularisation (Weier et. al 2021)
 *  on top of a unidirectional path tracer. The integrator requires materials to implement
 *  'eval_rough', 'pdf_rough' and 'sample_rough' which take the accumulated roughness as an
 *  additional parameter to evaluate the material with the updated roughness. Note that smooth dielectrics
 *  and smooth conductors are not directly supported and you should use their rough counterparts 
 *  (roughdielectric/roughconductor) while settimg the roughness to some epsilon value instead 
 *  (we typically use alpha=1e-4 or 1e-5 in our scenes to approximate perfectly specular materials).
 * 
 *  The algorithm keeps track of vertices in the path in a simple datastructure and computes the 
 *  accumulated roughness with 'get_roughening_opsr_interp' at every vertex before the material.
 *  is evaluated and sampled. The regularised contribution are then weighted using MIS while the path 
 *  prefix is constructed in an unbiased way (by sampling a new ray from the non-regularised bsdf). 
 *  (See Section 5.1 of our paper for more information)
 * 
 *  A faster, but slightly more biased version of the algorithm can also be used by setting
 *  'regularise_all' to true. This will effectively regularise all the interactions in a path and
 *  therefore also uses the regularised material to sample the path prefix, avoiding the somewhat costly
 *  extra ray casting.
 */
template <typename Float, typename Spectrum>
class PathOPSRInterpLearntIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth,
                    m_hide_emitters)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, Path, BSDF,
                     BSDFPtr)

    PathOPSRInterpLearntIntegrator(const Properties &props) : Base(props) {
        m_roughness_res = 4;

        m_roughening_mode = props.string("roughening_mode", "low"); 
        m_regularise_all = props.bool_("regularise_all", false);

        if (m_roughening_mode == "low") {
            m_att_factors_2d = DynamicBuffer<Float>::copy(
                att_factors_2d_interp_low.data(), att_factors_2d_interp_low.size());
            m_att_factors_3d = DynamicBuffer<Float>::copy(
                att_factors_3d_interp_low.data(), att_factors_3d_interp_low.size());
            m_att_factors_4d = DynamicBuffer<Float>::copy(
                att_factors_4d_interp_low.data(), att_factors_4d_interp_low.size());
            m_att_factors_5d = DynamicBuffer<Float>::copy(
                att_factors_5d_interp_low.data(), att_factors_5d_interp_low.size());
        } else if (m_roughening_mode == "moderate") {
            m_att_factors_2d = DynamicBuffer<Float>::copy(
                att_factors_2d_interp_moderate.data(), att_factors_2d_interp_moderate.size());
            m_att_factors_3d = DynamicBuffer<Float>::copy(
                att_factors_3d_interp_moderate.data(), att_factors_3d_interp_moderate.size());
            m_att_factors_4d = DynamicBuffer<Float>::copy(
                att_factors_4d_interp_moderate.data(), att_factors_4d_interp_moderate.size());
            m_att_factors_5d = DynamicBuffer<Float>::copy(
                att_factors_5d_interp_moderate.data(), att_factors_5d_interp_moderate.size());
        } else {
            Log(Error, "One of the roughening mode `low` or `moderate` must be chosen");
        }

        Log(Info, "%s", this);
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        RayDifferential3f ray = ray_;

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        // MIS weight for intersected emitters (set by prev. iteration)
        Float emission_weight(1.f);

        Spectrum throughput(1.f), result(0.f);

        // ---------------------- First intersection ----------------------

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        Mask valid_ray          = si.is_valid();
        EmitterPtr emitter      = si.emitter(scene);

        Path path(ray.o);

        // ---------------- Intersection with directly visible emitters ----------------

        if (any_or<true>(neq(emitter, nullptr)) && !m_hide_emitters){
            result[active] += emission_weight * throughput * emitter->eval(si, active);
        }

        for (int depth = 1;; ++depth) {

            active &= si.is_valid();

            /* Russian roulette: try to keep path weights equal to one,
            while accounting for the solid angle compression at refractive
            index boundaries. Stop with at least some probability to avoid
            getting stuck (e.g. due to total internal reflection) */
            if (depth > m_rr_depth) {
                Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                active &= sampler->next_1d(active) < q;
                throughput *= rcp(q);
            }

            // Stop if we've exceeded the number of requested bounces, or
            // if there are no more active lanes. Only do this latter check
            // in GPU mode when the number of requested bounces infinite
            // since it causes a costly synchronization.
            if ((uint32_t) depth >= (uint32_t) m_max_depth ||
                ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active)))
                break;

            BSDFContext ctx;
            BSDFPtr bsdf = si.bsdf(ray);

            /*
               We allow arbitrary bsdf to make a connection with a light but we
               will always use a smooth bsdf (no dirac delta function). So we
               should ideally throw an exception when any bsdf has a dirac delta
               distribution. But note that if any bsdf with a delta distribution
               is used in the scene it will never have a roughened contribution.
             */
            Mask active_e =
                active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            // Keep track of original bsdf roughness
            Vector2f bsdf_roughness = bsdf->get_roughness(si, active); 

            // Store the vertex and his roughness
            path.append_vertex(si.p, bsdf_roughness, active);

            // Calculate the accumulated roughness along the path. (Equation 14 in the paper)
            Vector2f acc_roughness = path.get_roughening_opsr_interp(
                m_att_factors_2d, m_att_factors_3d, m_att_factors_4d,
                m_att_factors_5d, m_roughness_res, active);

            // --------------------- Regularised Emitter sampling ---------------------

            if (likely(any_or<true>(active_e))) {
                auto [ds, emitter_val] = scene->sample_emitter_direction(
                    si, sampler->next_2d(active_e), true, active_e);
                active_e &= neq(ds.pdf, 0.f);

                // Query the BSDF for that emitter-sampled direction
                Vector3f wo = si.to_local(ds.d);

                Spectrum bsdf_val =
                    bsdf->eval_rough(ctx, si, wo, acc_roughness, active_e);
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Determine density of sampling that same direction using BSDF
                // sampling
                Float bsdf_pdf =
                    bsdf->pdf_rough(ctx, si, wo, acc_roughness, active_e);

                Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
                result[active_e] += mis * throughput * bsdf_val * emitter_val;
            }

            // ----------------------- Regularised BSDF sampling for MIS ----------------------

            // Sample regularised BSDF * cos(theta)
            auto [bs_rough, bsdf_val_rough] = bsdf->sample_rough(
                ctx, si, sampler->next_1d(active), sampler->next_2d(active),
                acc_roughness, active);
            
            // Set the sampled type of bsdf
            path.set_sampled_component_for_last(bs_rough.sampled_type);

            bsdf_val_rough = si.to_world_mueller(bsdf_val_rough, -bs_rough.wo, si.wi);

            Mask active_bsdf_val = active & any(neq(depolarize(bsdf_val_rough), 0.f));
            
            // If any of the bsdf evaluation is non-zero perform MIS weighting
            // with a possible emitter intersection 
            if (any_or<true>(active_bsdf_val)) {
             
                // Intersect the regularised BSDF ray against the scene geometry
                ray = si.spawn_ray(si.to_world(bs_rough.wo));
                SurfaceInteraction3f si_bsdf =
                    scene->ray_intersect(ray, active);

                /* Determine probability of having sampled that same
                    direction using emitter sampling. */
                emitter = si_bsdf.emitter(scene, active);
                DirectionSample3f ds(si_bsdf, si);
                ds.object = emitter;

                if (any_or<true>(neq(emitter, nullptr))) {
                    Float emitter_pdf = select(
                        neq(emitter, nullptr) &&
                            !has_flag(bs_rough.sampled_type, BSDFFlags::Delta),
                        scene->pdf_emitter_direction(si, ds), 0.f);

                    emission_weight = mis_weight(bs_rough.pdf, emitter_pdf);
                    // Add the MIS weighted contribution
                    result[active] += emission_weight * throughput *
                                      bsdf_val_rough *
                                      emitter->eval(si_bsdf, active);
                }

                // If the roughness of the material is identical (e.g. for
                // direct illumination) we can reuse the sample without
                // introducing any bias.
                Mask active_equal_roughness = active & any(neq(bsdf_roughness, acc_roughness));

                if (none_or<false>(active_equal_roughness) || m_regularise_all) {
                    throughput = throughput * bsdf_val_rough;

                    active &= any(neq(depolarize(throughput), 0.f));

                    if (none_or<false>(active))
                        break;

                    eta *= bs_rough.eta;

                    si = std::move(si_bsdf);

                    // Continue to the next iteration as this was equivalent to
                    // a non-regularised sampling
                    continue;
                }
            }else if (m_regularise_all){
                // Fast path if all bsdfs have 0 contribution and we always roughen
                break;
            }

            // ----------------------- Regular BSDF sampling for next intersection ----------------------
       
            // Sample BSDF * cos(theta)
            auto [bs, bsdf_val] = bsdf->sample(
                ctx, si, sampler->next_1d(active), sampler->next_2d(active), active);

            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            throughput = throughput * bsdf_val;
            active &= any(neq(depolarize(throughput), 0.f));
            if (none_or<false>(active))
                break;

            eta *= bs.eta;

            // Intersect the unbiased BSDF ray against the scene geometry
            ray = si.spawn_ray(si.to_world(bs.wo));
            si  = scene->ray_intersect(ray, active);
        }

        return { result, valid_ray };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathOPSRInterpLearntIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "  roughness_res = %i\n"
                           "  roughening_mode = %s\n"
                           "  regularise_all = %i\n"
                           "]",
                           m_max_depth, m_rr_depth, m_roughness_res,
                           m_roughening_mode, m_regularise_all);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    MTS_DECLARE_CLASS()

protected: 
    std::string m_roughening_mode;
    size_t m_roughness_res;
    DynamicBuffer<Float> m_att_factors_2d;
    DynamicBuffer<Float> m_att_factors_3d;
    DynamicBuffer<Float> m_att_factors_4d;
    DynamicBuffer<Float> m_att_factors_5d;
    bool m_regularise_all;
};

MTS_IMPLEMENT_CLASS_VARIANT(PathOPSRInterpLearntIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PathOPSRInterpLearntIntegrator, "OPSR Interp Path tracer integrator");
NAMESPACE_END(mitsuba)
