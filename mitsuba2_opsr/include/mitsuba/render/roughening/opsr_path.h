
#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/render/bsdf.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum> class Path {
public:
    MTS_IMPORT_TYPES(BSDF, BSDFPtr)

    struct PathVertex {

        /* Component type of the underlying BDSF that was sampled, 0 if an
         * emitter or a sensor*/
        UInt32 sampled_component;

        Point3f position;

        Vector2f roughness;

        Mask active;

        PathVertex(Point3f position, Vector2f vertex_roughness, Mask active)
            : sampled_component(0), position(position),
              roughness(vertex_roughness), active(active) {}
    };

    /* A path is constructed using the ray origin but this origin doesn't count
     * as a vertex for the roughening */
    Path(const Point3f &origin) : m_origin(origin) {}

    void append_vertex(const Point3f &position,
                       const Vector2f &vertex_roughness, Mask active) {
        m_vertices.push_back(PathVertex(position, vertex_roughness, active));
    }

    void set_sampled_component_for_last(UInt32 sampled_component) {
        m_vertices.back().sampled_component = sampled_component;
    }

    /* Recursively computes the accumulated roughness along a path of arbitrary depth. (Equation 14 in the paper) */
    Vector2f get_roughening_opsr(const DynamicBuffer<Float> &att_factors_2d,
                                    const DynamicBuffer<Float> &att_factors_3d,
                                    const DynamicBuffer<Float> &att_factors_4d,
                                    const DynamicBuffer<Float> &att_factors_5d,
                                    size_t roughness_res, Mask active) {
        // quick path for direct illumination
        if (m_vertices.size() == 1)
            return m_vertices.back().roughness;

        // Denotes the number of vertices already processed
        size_t processed_idx = 0;

        // Query the first subpath attenuation
        Float attenuation_factor = get_roughening_attenuation_factor_5d(
            roughness_res, m_vertices.size(), 0, Vector2f(-1.0f),
            att_factors_2d, att_factors_3d, att_factors_4d, att_factors_5d,
            active);

        // Keep the first vertex in the path separately to update it with the
        // virtual roughness for longer paths
        Vector2f virtual_roughness = m_vertices[0].roughness;

        // Recursively estimate the attenuated roughening in the path
        while (true) {
            Vector2f accumulated_roughness = Vector2f(1.0f);

            if ((m_vertices.size() - processed_idx) <= 5) // base case
            {
                // Accumulate the roughness with a potential virtual roughness
                accumulated_roughness *=
                    (1.0f - attenuation_factor * virtual_roughness);
                for (size_t i = processed_idx + 1; i < m_vertices.size() - 1;
                     i++) {
                    accumulated_roughness *=
                        (1.0f - attenuation_factor * m_vertices[i].roughness);
                }
                accumulated_roughness *= (1.0f - m_vertices.back().roughness);

                return 1.0f - accumulated_roughness;
            } else {
                // accumulate 5 vertex and update path roughness with updated
                // virtual roughness
                const size_t last_idx = processed_idx + 4;
                for (size_t i = processed_idx; i < last_idx; i++) {
                    accumulated_roughness *=
                        (1.0f - attenuation_factor * m_vertices[i].roughness);
                }
                accumulated_roughness *=
                    (1.0f - m_vertices[last_idx].roughness);

                // update processed index (5 - the virtual roughness updated)
                processed_idx += 4;

                // update the virtual roughness of the new subpath
                virtual_roughness = (1.0f - accumulated_roughness);

                // query next attenuation factor
                attenuation_factor = get_roughening_attenuation_factor_5d(
                    roughness_res, m_vertices.size() - processed_idx, last_idx,
                    (1.0f - accumulated_roughness), att_factors_2d,
                    att_factors_3d, att_factors_4d, att_factors_5d, active);
            }
        }
    }

    /* Recursively computes the accumulated roughness along a path of arbitrary depth. (Equation 14 in the paper)
       This method also uses a 5-dimensional interpolation when looking up the attenuation factors, further improving
       the robustness during optimisation and avoiding discontinuities. */
    Vector2f
    get_roughening_opsr_interp(const DynamicBuffer<Float> &att_factors_2d,
                                  const DynamicBuffer<Float> &att_factors_3d,
                                  const DynamicBuffer<Float> &att_factors_4d,
                                  const DynamicBuffer<Float> &att_factors_5d,
                                  size_t roughness_res, Mask active) {
        // quick path for direct illumination or path ending with diffuse
        // roughness
        if (m_vertices.size() == 1)
            return m_vertices.back().roughness;

        if (roughness_to_bin(m_vertices.back().roughness, roughness_res) ==
            UInt32(roughness_res - 1)) {
            return m_vertices.back().roughness;
        }

        // Denotes the number of vertices already processed
        size_t processed_idx = 0;

        // Query the first subpath attenuation
        Float attenuation_factor = get_roughening_attenuation_factor_5d_interp(
            roughness_res, m_vertices.size(), 0, Vector2f(-1.0f),
            att_factors_2d, att_factors_3d, att_factors_4d, att_factors_5d,
            active);

        // Keep the first vertex in the path separately to update it with the
        // virtual roughness for longer paths
        Vector2f virtual_roughness = m_vertices[0].roughness;

        // Recursively estimate the attenuated roughening in the path
        while (true) {
            Vector2f accumulated_roughness = Vector2f(1.0f);

            if ((m_vertices.size() - processed_idx) <= 5) // base case
            {
                // Accumulate the roughness with a potential virtual roughness
                accumulated_roughness *=
                    (1.0f - attenuation_factor * virtual_roughness);
                for (size_t i = processed_idx + 1; i < m_vertices.size() - 1;
                     i++) {
                    accumulated_roughness *=
                        (1.0f - attenuation_factor * m_vertices[i].roughness);
                }
                accumulated_roughness *= (1.0f - m_vertices.back().roughness);

                return 1.0f - accumulated_roughness;
            } else {
                // accumulate 5 vertex and update path roughness with updated
                // virtual roughness
                const size_t last_idx = processed_idx + 4;
                accumulated_roughness *=
                    (1.0f - attenuation_factor * virtual_roughness);
                for (size_t i = processed_idx + 1; i < last_idx; i++) {
                    accumulated_roughness *=
                        (1.0f - attenuation_factor * m_vertices[i].roughness);
                }
                accumulated_roughness *=
                    (1.0f - m_vertices[last_idx].roughness);

                // update processed index (5 - the virtual roughness updated)
                processed_idx += 4;

                // update the virtual roughness of the new subpath
                virtual_roughness = (1.0f - accumulated_roughness);

                // query next attenuation factor
                attenuation_factor =
                    get_roughening_attenuation_factor_5d_interp(
                        roughness_res, m_vertices.size() - processed_idx,
                        last_idx, virtual_roughness, att_factors_2d,
                        att_factors_3d, att_factors_4d, att_factors_5d, active);
            }
        }
    }

    /* Computes the accumulated roughness along a path for a given attenuation factor */
    Vector2f get_accumulated_roughness(Float attenuation_factor) {
        Vector2f accumulated_roughness = 1.0f;
        Vector2f curr_vertex_roughness = m_vertices.back().roughness;
        accumulated_roughness *= (1.0f - curr_vertex_roughness);

        for (size_t i = 0; i < m_vertices.size() - 1; i++) {
            accumulated_roughness *=
                (1.0f - m_vertices[i].roughness * attenuation_factor);
        }
        return 1.0f - accumulated_roughness;
    }

    size_t length() const { return m_vertices.size(); }

    Point3f origin() const { return m_origin; }

    std::vector<PathVertex> vertices() const { return m_vertices; }

private:
    // Fast integer exponentiation
    // https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
    int ipow(int base, int exp) {
        int result = 1;
        for (;;) {
            if (exp & 1)
                result *= base;
            exp >>= 1;
            if (!exp)
                break;
            base *= base;
        }
        return result;
    }

    // Conversion from roughness alpha to the bin's index i with Quantisation level Q. 
    // (See Section 4.4 : Discrete path space classification)
    UInt32 roughness_to_bin(const Vector2f &vertex_roughness,
                            size_t roughness_res) {
        // Uses a logarithmis scale discretisation
        size_t log_roughness_res = roughness_res + 1;
        return (UInt32) min(
            floor((pow(2.0f, sqrt(hmin(vertex_roughness))) - 1.0f) *
                  log_roughness_res),
            roughness_res - 1);
    }

    // 5-dimensional interpolated lookup of the attenuation factor
    Float get_roughening_attenuation_factor_5d(
        size_t roughness_res, size_t size, size_t start_idx,
        const Vector2f &virtual_roughness,
        const DynamicBuffer<Float> &att_factors_2d,
        const DynamicBuffer<Float> &att_factors_3d,
        const DynamicBuffer<Float> &att_factors_4d,
        const DynamicBuffer<Float> &att_factors_5d, Mask active) {

        const size_t max_subpath_len = 5;
        const size_t subpath_len     = std::min(size, max_subpath_len);
        UInt32 query_idx             = 0;

        for (size_t i = start_idx; i < start_idx + subpath_len; i++) {
            if (i == start_idx && virtual_roughness != Vector2f(-1.0f)) {
                // Use the virtual roughness instead of the start_idx for the
                // first vertex if a virtual roughness was accumulated
                query_idx += roughness_to_bin(virtual_roughness, roughness_res);
            } else {
                query_idx +=
                    ipow(roughness_res, (i - start_idx)) *
                    roughness_to_bin(m_vertices[i].roughness, roughness_res);
            }
        }

        Float attenuation_factor = 0.0f;
        switch (subpath_len) {
            case 2:
                attenuation_factor =
                    gather<Float>(att_factors_2d, query_idx, active);
                break;
            case 3:
                attenuation_factor =
                    gather<Float>(att_factors_3d, query_idx, active);
                break;
            case 4:
                attenuation_factor =
                    gather<Float>(att_factors_4d, query_idx, active);
                break;
            case 5:
                attenuation_factor =
                    gather<Float>(att_factors_5d, query_idx, active);
                break;
            default:
                printf("%zu", subpath_len);
        }
        return attenuation_factor;
    }

    // Conversion from roughness alpha to the bin's index i with Quantisation level Q. 
    // (fractional variant for interpolated lookup)
    // (See Section 4.4 : Discrete path space classification)
    Float roughness_to_fractional_bin(const Vector2f &vertex_roughness,
                                      size_t roughness_res) {
        size_t log_roughness_res = roughness_res + 1;
        return min((pow(2.0f, sqrt(hmin(vertex_roughness))) - 1.0f) *
                       log_roughness_res,
                   roughness_res - 1);
    }

    // 5-dimensional interpolated lookup of the attenuation factor
    Float get_roughening_attenuation_factor_5d_interp(
        size_t roughness_res, size_t size, size_t start_idx,
        const Vector2f &virtual_roughness,
        const DynamicBuffer<Float> &att_factors_2d,
        const DynamicBuffer<Float> &att_factors_3d,
        const DynamicBuffer<Float> &att_factors_4d,
        const DynamicBuffer<Float> &att_factors_5d, Mask active) {

        const size_t max_subpath_len = 5;
        const size_t subpath_len     = std::min(size, max_subpath_len);

        std::vector<std::pair<UInt32, UInt32>> bins(subpath_len, { 0, 0 });
        std::vector<Float> fractional_bins(subpath_len, 0.0);

        for (size_t i = start_idx; i < start_idx + subpath_len; i++) {
            Float frac_bin;
            if (i == start_idx && virtual_roughness != Vector2f(-1.0f)) {
                // Use the virtual roughness instead of the start_idx for the
                // first vertex if a virtual roughness was accumulated
                frac_bin = roughness_to_fractional_bin(virtual_roughness,
                                                       roughness_res);
            } else {
                frac_bin = roughness_to_fractional_bin(m_vertices[i].roughness,
                                                       roughness_res);
            }
            bins[i - start_idx].first = (UInt32) floor(frac_bin);
            bins[i - start_idx].second =
                (UInt32) min(ceil(frac_bin), roughness_res - 1);
            fractional_bins[i - start_idx] =
                frac_bin - bins[i - start_idx].first;
        }

        size_t num_combinations = ipow(2, subpath_len);
        std::vector<UInt32> query_indices(num_combinations, 0);

        for (size_t c = 0; c < num_combinations; c++) {
            UInt32 query_idx = 0;
            for (size_t i = 0; i < subpath_len; i++) {
                UInt32 bin_index =
                    (c >> i & 0b1) == 0 ? bins[i].first : bins[i].second;
                query_idx += ipow(roughness_res, i) * bin_index;
            }
            query_indices[c] = query_idx;
        }

        const DynamicBuffer<Float> *att_factors = nullptr;

        switch (subpath_len) {
            case 2:
                att_factors = &att_factors_2d;
                break;
            case 3:
                att_factors = &att_factors_3d;
                break;
            case 4:
                att_factors = &att_factors_4d;
                break;
            case 5:
                att_factors = &att_factors_5d;
                break;
            default:
                printf("%zu", subpath_len);
        }

        // Perform an n-dimensional linear interpolation
        Float attenuation_factor = 0.0;
        for (size_t c = 0; c < num_combinations; c++) {
            Float att_value =
                gather<Float>(*att_factors, query_indices[c], active);
            Float ratio = 1.0;
            for (size_t i = 0; i < subpath_len; i++) {
                ratio *= (c >> i & 0b1) == 0 ? (Float(1.0) - fractional_bins[i])
                                             : fractional_bins[i];
            }
            attenuation_factor += att_value * ratio;
        }

        return attenuation_factor;
    }

    std::vector<PathVertex> m_vertices;
    Point3f m_origin;
};

NAMESPACE_END(mitsuba)