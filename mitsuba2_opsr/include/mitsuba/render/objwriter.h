#pragma once

#include <fstream>
#include <mitsuba/core/math.h>
#include <mitsuba/core/string.h>
#include <mitsuba/mitsuba.h>

NAMESPACE_BEGIN(mitsuba)

/**
 *  A Simple obj writer for convenience which only works with the scalar
 *  variants!
 *
 *  Be sure to wrap the call in a : if constexpr (is_scalar_v<Float>)
 *  statement
 **/

#define DEBUG

struct OBJWriter {
public:
    using Float = float;
    MTS_IMPORT_CORE_TYPES()

    OBJWriter()
        : faces_offset(0), lines_offset(0), vertices_offset(0),
          object_offset(0) {}

    // Opens an obj file and sets offsets accordingly
    OBJWriter(std::string filename) : filename(filename) {
        std::ifstream file(filename + ".obj");
        faces_offset    = 0;
        lines_offset    = 0;
        vertices_offset = 0;
        object_offset   = 0;
        if (file.good()) {

            std::string line;
            while (getline(file, line)) {
                if (string::starts_with(line, "v"))
                    vertices_offset++;
                if (string::starts_with(line, "l"))
                    lines_offset++;
                if (string::starts_with(line, "f"))
                    faces_offset++;
                if (string::starts_with(line, "o"))
                    object_offset++;
            }
        }

        file.close();
    }

    void add_line(const Point3f &p0, const Point3f &p1) {
        vertices.push_back(p0);
        vertices.push_back(p1);
        lines.push_back(Vector2i(vertices.size(), vertices.size() - 1));
    }

    void add_box(const Point3f &min, const Point3f &max) {
        Point3f p000(min.x(), min.y(), min.z());
        Point3f p100(max.x(), min.y(), min.z());
        Point3f p010(min.x(), max.y(), min.z());
        Point3f p110(max.x(), max.y(), min.z());
        Point3f p001(min.x(), min.y(), max.z());
        Point3f p101(max.x(), min.y(), max.z());
        Point3f p011(min.x(), max.y(), max.z());
        Point3f p111(max.x(), max.y(), max.z());
        vertices.push_back(p000); // 7
        vertices.push_back(p100); // 6
        vertices.push_back(p010); // 5
        vertices.push_back(p110); // 4
        vertices.push_back(p001); // 3
        vertices.push_back(p101); // 2
        vertices.push_back(p011); // 1
        vertices.push_back(p111); // 0
        size_t id = vertices.size();
        lines.push_back(Vector2i(id - 7, id - 6));
        lines.push_back(Vector2i(id - 6, id - 4));
        lines.push_back(Vector2i(id - 4, id - 5));
        lines.push_back(Vector2i(id - 5, id - 7));
        lines.push_back(Vector2i(id - 3, id - 2));
        lines.push_back(Vector2i(id - 2, id));
        lines.push_back(Vector2i(id, id - 1));
        lines.push_back(Vector2i(id - 1, id - 3));
        lines.push_back(Vector2i(id - 7, id - 3));
        lines.push_back(Vector2i(id - 6, id - 2));
        lines.push_back(Vector2i(id - 5, id - 1));
        lines.push_back(Vector2i(id - 4, id));
    }
    // p0 and p1 form the center line of the cylinder
    void add_cylinder(const Point3f &p0, const Point3f &p1, Float radius,
                      size_t res) {

        Float step = 2.0f * math::Pi<Float> / res;

        // Center vertices
        vertices.push_back(p0);
        vertices.push_back(p1);
        size_t id_c0 = vertices.size() - 1;
        size_t id_c1 = vertices.size();

        Point3f tmp_p0(0.f), tmp_p1(0.f);
        Vector3f direction = normalize(p1 - p0);

        // Starting vertices
        size_t id_s0 = vertices.size() + 1;
        size_t id_s1 = vertices.size() + 2;

        size_t id_tmp0 = id_s0;
        size_t id_tmp1 = id_s1;

        for (size_t i = 0; i <= res; i++) {

            float t     = step * i;
            auto [s, c] = sincos(t);

            tmp_p0 = Point3f(radius * c, radius * s, 0.0f);
            tmp_p1 = Point3f(radius * c, radius * s, 0.0f);

            // Calculate necessary rotation to aligned disks with direction
            Vector3f rotation_axis = normalize(cross(tmp_p0, direction));
            Float angle            = dot(normalize(tmp_p0), direction);
            Transform4f rot =
                Transform4f().rotate(rotation_axis, rad_to_deg(-angle));

            size_t id_n0, id_n1;
            if (i == res) {
                id_n0 = id_s0;
                id_n1 = id_s1;
            } else {
                vertices.push_back(Transform4f().translate(p0) * rot * tmp_p0);
                vertices.push_back(Transform4f().translate(p1) * rot * tmp_p1);

                id_n0 = vertices.size() - 1;
                id_n1 = vertices.size();
            }
            if (i > 0) {
                // core part
                faces.push_back(Vector3i(id_tmp1, id_tmp0, id_n0));
                faces.push_back(Vector3i(id_tmp1, id_n0, id_n1));
                // top cap
                faces.push_back(Vector3i(id_tmp1, id_n1, id_c1));
                // bottom cap
                faces.push_back(Vector3i(id_tmp0, id_c0, id_n0));
                // update for next iteration
                id_tmp0 = id_n0;
                id_tmp1 = id_n1;
            }
        }
    }

    void add_face(const Point3f &p0, const Point3f &p1, const Point3f &p2) {
        vertices.push_back(p0);
        vertices.push_back(p1);
        vertices.push_back(p2);
        faces.push_back(Vector3i(vertices.size(), vertices.size() - 1,
                                 vertices.size() - 2));
    }

    void add_quad(const Point3f &p0, const Point3f &p1, const Point3f &p2,
                  const Point3f &p3) {

        vertices.push_back(p0);
        vertices.push_back(p1);
        vertices.push_back(p2);
        vertices.push_back(p3);
        size_t id = vertices.size();
        faces.push_back(Vector3i(id - 1, id, id - 2));
        faces.push_back(Vector3i(id - 1, id - 3, id));
    }

    void write() {
        if (filename.empty()) {
            Log(Warn, "Obj file not written please specify a filename first!");
        } else {
            append(filename);
        }
    }

    void write(std::string name, bool smooth = false) const {
#ifdef DEBUG
        Log(Info, "Writing ", name + ".obj");
#endif
        std::ofstream file(name + ".obj");
        if (file.is_open()) {
            file << "# Saved from Mitsuba2\n";
            file << "o " << name << "\n";
            for (size_t v = 0; v < vertices.size(); v++) {
                file << "v " << vertices[v].x() << " " << vertices[v].y() << " "
                     << vertices[v].z() << "\n";
            }
            if (!smooth)
                file << "s off"
                     << "\n";
            for (size_t l = 0; l < lines.size(); l++) {
                file << "l " << lines[l].x() << " " << lines[l].y() << "\n";
            }
            for (size_t f = 0; f < faces.size(); f++) {
                file << "f " << faces[f].x() << " " << faces[f].y() << " "
                     << faces[f].z() << "\n";
            }
            file << "\n";
        } else {
            Log(Warn, "Unable to open file ", name + ".obj");
        }
    }

private:
    void append(std::string name, bool smooth = false) const {
        std::ifstream temp_file(name + ".obj");
        // fallback to write if file doesn't exist already
        if (!temp_file.good()) {
            temp_file.close();
            write(name);
        } else {
            temp_file.close();
#ifdef DEBUG
            Log(Info, "Appending to ", name + ".obj");
#endif
            std::ofstream file(name + ".obj", std::ios_base::app);
            if (file.is_open()) {
                file << "o " << name << "_" << std::to_string(object_offset)
                     << "\n";
                for (size_t v = 0; v < vertices.size(); v++) {
                    file << "v " << vertices[v].x() << " " << vertices[v].y()
                         << " " << vertices[v].z() << "\n";
                }
                if (!smooth)
                    file << "s off"
                         << "\n";
                for (size_t l = 0; l < lines.size(); l++) {
                    file << "l " << lines[l].x() + vertices_offset << " "
                         << lines[l].y() + vertices_offset << "\n";
                }
                for (size_t f = 0; f < faces.size(); f++) {
                    file << "f " << faces[f].x() + vertices_offset << " "
                         << faces[f].y() + vertices_offset << " "
                         << faces[f].z() + vertices_offset << "\n";
                }
            } else {
                Log(Warn, "Unable to open file ", name + ".obj");
            }
        }
    }

    std::vector<Point3f> vertices;
    std::vector<Vector2i> lines;
    std::vector<Vector3i> faces;
    size_t faces_offset;
    size_t lines_offset;
    size_t vertices_offset;
    size_t object_offset;
    std::string filename;
};

NAMESPACE_END(mitsuba)