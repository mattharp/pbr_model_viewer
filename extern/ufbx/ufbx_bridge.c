/*
 * ufbx_bridge.c
 * Thin C bridge over ufbx for Python ctypes consumption.
 *
 * Compile:
 *   gcc -shared -O2 -o ufbx.dll ufbx.c ufbx_bridge.c -DUFBX_BRIDGE_EXPORT -lm
 *   (on Linux: replace .dll with .so, add -fPIC)
 */

#define UFBX_BRIDGE_EXPORT
#include "ufbx_bridge.h"
#include "ufbx.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Helpers ---- */

static ufbx_scene *to_scene(fbx_scene_handle h) {
    return (ufbx_scene *)h;
}

/* Map our bridge map-type enum to ufbx PBR map indices */
static ufbx_material_pbr_map bridge_map_to_ufbx(int map_type) {
    switch (map_type) {
        case FBX_MAP_BASE_COLOR: return UFBX_MATERIAL_PBR_BASE_COLOR;
        case FBX_MAP_METALLIC:   return UFBX_MATERIAL_PBR_METALNESS;
        case FBX_MAP_ROUGHNESS:  return UFBX_MATERIAL_PBR_ROUGHNESS;
        case FBX_MAP_NORMAL:     return UFBX_MATERIAL_PBR_NORMAL_MAP;
        case FBX_MAP_EMISSIVE:   return UFBX_MATERIAL_PBR_EMISSION_COLOR;
        case FBX_MAP_OCCLUSION:  return UFBX_MATERIAL_PBR_AMBIENT_OCCLUSION;
        default: return UFBX_MATERIAL_PBR_BASE_COLOR;
    }
}

/* Get the ufbx_texture pointer for a PBR map slot, or NULL */
static ufbx_texture *get_pbr_texture(ufbx_material *mat, int map_type) {
    if (!mat) return NULL;
    ufbx_material_pbr_map pbr_idx = bridge_map_to_ufbx(map_type);
    ufbx_material_map *m = &mat->pbr.maps[pbr_idx];
    if (m->texture_enabled && m->texture) {
        return m->texture;
    }
    return NULL;
}

/* ---- Scene loading / freeing ---- */

BRIDGE_API fbx_scene_handle fbx_load(const char *filepath) {
    ufbx_load_opts opts = {0};
    /* Generate normals if missing */
    opts.generate_missing_normals = true;
    /* Triangulate in-place for simpler extraction */
    opts.target_axes = ufbx_axes_right_handed_y_up;
    opts.target_unit_meters = 1.0f;

    ufbx_error error;
    ufbx_scene *scene = ufbx_load_file(filepath, &opts, &error);
    if (!scene) {
        fprintf(stderr, "ufbx_bridge: failed to load '%s': %s\n",
                filepath, error.description.data);
        return NULL;
    }
    printf("ufbx_bridge: loaded '%s' (%zu meshes, %zu materials)\n",
           filepath, scene->meshes.count, scene->materials.count);
    return (fbx_scene_handle)scene;
}

BRIDGE_API void fbx_free(fbx_scene_handle scene) {
    if (scene) {
        ufbx_free_scene(to_scene(scene));
    }
}

/* ---- Scene queries ---- */

BRIDGE_API int fbx_mesh_count(fbx_scene_handle scene) {
    return scene ? (int)to_scene(scene)->meshes.count : 0;
}

BRIDGE_API int fbx_material_count(fbx_scene_handle scene) {
    return scene ? (int)to_scene(scene)->materials.count : 0;
}

/* ---- Mesh data extraction ---- */

/*
 * Internal vertex representation matching our VBO layout:
 * position(3) + normal(3) + uv(2) + tangent(3) = 11 floats
 */
typedef struct {
    float pos[3];
    float norm[3];
    float uv[2];
    float tan[3];
} bridge_vertex;

BRIDGE_API void fbx_mesh_get_counts(fbx_scene_handle scene, int mesh_idx,
                                    int *out_vertex_count, int *out_index_count)
{
    *out_vertex_count = 0;
    *out_index_count = 0;

    ufbx_scene *s = to_scene(scene);
    if (!s || mesh_idx < 0 || (size_t)mesh_idx >= s->meshes.count) return;

    ufbx_mesh *mesh = s->meshes.data[mesh_idx];

    /* Count total triangles across all faces */
    int total_tris = 0;
    for (size_t i = 0; i < mesh->faces.count; i++) {
        ufbx_face face = mesh->faces.data[i];
        if (face.num_indices >= 3) {
            total_tris += (int)(face.num_indices - 2);
        }
    }

    /* Worst case: 3 unique vertices per triangle */
    *out_vertex_count = total_tris * 3;
    *out_index_count = total_tris * 3;
}

BRIDGE_API int fbx_mesh_extract(fbx_scene_handle scene, int mesh_idx,
                                float *out_vertices, uint32_t *out_indices,
                                int max_vertices, int max_indices)
{
    ufbx_scene *s = to_scene(scene);
    if (!s || mesh_idx < 0 || (size_t)mesh_idx >= s->meshes.count) return 0;

    ufbx_mesh *mesh = s->meshes.data[mesh_idx];

    /* Allocate temporary vertex buffer */
    int vert_cap = max_vertices;
    bridge_vertex *verts = (bridge_vertex *)calloc(vert_cap, sizeof(bridge_vertex));
    if (!verts) return 0;

    int num_verts = 0;

    /* Temporary buffer for triangulation indices */
    size_t tri_buf_size = mesh->max_face_triangles * 3;
    uint32_t *tri_indices = (uint32_t *)calloc(tri_buf_size, sizeof(uint32_t));
    if (!tri_indices) { free(verts); return 0; }

    /* Iterate all faces and triangulate */
    for (size_t fi = 0; fi < mesh->faces.count; fi++) {
        ufbx_face face = mesh->faces.data[fi];

        uint32_t num_tris = ufbx_triangulate_face(tri_indices, tri_buf_size, mesh, face);

        for (size_t ti = 0; ti < (size_t)(num_tris * 3); ti++) {
            uint32_t index = tri_indices[ti];

            if (num_verts >= vert_cap) break;

            bridge_vertex *v = &verts[num_verts];

            /* Position */
            ufbx_vec3 pos = ufbx_get_vertex_vec3(&mesh->vertex_position, index);
            v->pos[0] = (float)pos.x;
            v->pos[1] = (float)pos.y;
            v->pos[2] = (float)pos.z;

            /* Normal */
            if (mesh->vertex_normal.exists) {
                ufbx_vec3 n = ufbx_get_vertex_vec3(&mesh->vertex_normal, index);
                v->norm[0] = (float)n.x;
                v->norm[1] = (float)n.y;
                v->norm[2] = (float)n.z;
            } else {
                v->norm[0] = 0.0f; v->norm[1] = 1.0f; v->norm[2] = 0.0f;
            }

            /* UV */
            if (mesh->vertex_uv.exists) {
                ufbx_vec2 uv = ufbx_get_vertex_vec2(&mesh->vertex_uv, index);
                v->uv[0] = (float)uv.x;
                v->uv[1] = (float)uv.y;
            } else {
                v->uv[0] = 0.0f; v->uv[1] = 0.0f;
            }

            /* Tangent */
            if (mesh->vertex_tangent.exists) {
                ufbx_vec3 t = ufbx_get_vertex_vec3(&mesh->vertex_tangent, index);
                v->tan[0] = (float)t.x;
                v->tan[1] = (float)t.y;
                v->tan[2] = (float)t.z;
            } else {
                v->tan[0] = 1.0f; v->tan[1] = 0.0f; v->tan[2] = 0.0f;
            }

            num_verts++;
        }
    }

    /* Deduplicate vertices using ufbx_generate_indices */
    ufbx_vertex_stream streams[1] = {
        { verts, (size_t)num_verts, sizeof(bridge_vertex) }
    };

    size_t num_indices = (size_t)num_verts;
    if (num_indices > (size_t)max_indices) num_indices = (size_t)max_indices;

    size_t unique_count = ufbx_generate_indices(streams, 1, out_indices,
                                                 num_indices, NULL, NULL);

    /* Copy deduplicated vertices into output (11 floats each) */
    int out_count = (int)unique_count;
    if (out_count > max_vertices) out_count = max_vertices;

    memcpy(out_vertices, verts, out_count * sizeof(bridge_vertex));

    free(tri_indices);
    free(verts);

    return out_count;
}

BRIDGE_API int fbx_mesh_material_index(fbx_scene_handle scene, int mesh_idx) {
    ufbx_scene *s = to_scene(scene);
    if (!s || mesh_idx < 0 || (size_t)mesh_idx >= s->meshes.count) return -1;

    ufbx_mesh *mesh = s->meshes.data[mesh_idx];
	if (mesh->materials.count > 0 && mesh->materials.data[0]) {
		return (int)mesh->materials.data[0]->element.typed_id;
    }
    return -1;
}

/* ---- Material queries ---- */

BRIDGE_API void fbx_material_get_info(fbx_scene_handle scene, int mat_idx,
                                      fbx_material_info *out)
{
    memset(out, 0, sizeof(*out));
    /* Defaults */
    out->base_color[0] = 0.7f; out->base_color[1] = 0.6f;
    out->base_color[2] = 0.5f; out->base_color[3] = 1.0f;
    out->roughness = 0.5f;

    ufbx_scene *s = to_scene(scene);
    if (!s || mat_idx < 0 || (size_t)mat_idx >= s->materials.count) return;

    ufbx_material *mat = s->materials.data[mat_idx];

    /* Base color */
    ufbx_material_map *bc = &mat->pbr.base_color;
    if (bc->has_value) {
        out->base_color[0] = (float)bc->value_vec4.x;
        out->base_color[1] = (float)bc->value_vec4.y;
        out->base_color[2] = (float)bc->value_vec4.z;
        out->base_color[3] = (float)bc->value_vec4.w;
    }

    /* Metallic */
    ufbx_material_map *met = &mat->pbr.metalness;
    if (met->has_value) {
        out->metallic = (float)met->value_real;
    }

    /* Roughness */
    ufbx_material_map *rough = &mat->pbr.roughness;
    if (rough->has_value) {
        out->roughness = (float)rough->value_real;
    }

    /* Emissive */
    ufbx_material_map *em = &mat->pbr.emission_color;
    if (em->has_value) {
        out->emissive[0] = (float)em->value_vec4.x;
        out->emissive[1] = (float)em->value_vec4.y;
        out->emissive[2] = (float)em->value_vec4.z;
    }

    /* Check which maps have textures */
    for (int i = 0; i < FBX_MAP_COUNT; i++) {
        out->has_texture[i] = (get_pbr_texture(mat, i) != NULL) ? 1 : 0;
    }
}

/* ---- Texture extraction ---- */

BRIDGE_API int fbx_material_get_texture(fbx_scene_handle scene, int mat_idx,
                                        int map_type, fbx_texture_info *out)
{
    memset(out, 0, sizeof(*out));

    ufbx_scene *s = to_scene(scene);
    if (!s || mat_idx < 0 || (size_t)mat_idx >= s->materials.count) return 0;

    ufbx_material *mat = s->materials.data[mat_idx];
    ufbx_texture *tex = get_pbr_texture(mat, map_type);
    if (!tex) return 0;

    /* Check for embedded content */
    if (tex->content.size > 0) {
        out->data = tex->content.data;
        out->size = (uint32_t)tex->content.size;
    }

    /* Copy filename hint */
    if (tex->relative_filename.length > 0) {
        size_t len = tex->relative_filename.length;
        if (len >= sizeof(out->filename)) len = sizeof(out->filename) - 1;
        memcpy(out->filename, tex->relative_filename.data, len);
        out->filename[len] = '\0';
    } else if (tex->filename.length > 0) {
        size_t len = tex->filename.length;
        if (len >= sizeof(out->filename)) len = sizeof(out->filename) - 1;
        memcpy(out->filename, tex->filename.data, len);
        out->filename[len] = '\0';
    }

    return (out->data != NULL && out->size > 0) ? 1 : 0;
}

BRIDGE_API int fbx_material_get_texture_path(fbx_scene_handle scene, int mat_idx,
                                             int map_type, char *out_path, int max_len)
{
    ufbx_scene *s = to_scene(scene);
    if (!s || mat_idx < 0 || (size_t)mat_idx >= s->materials.count) return 0;

    ufbx_material *mat = s->materials.data[mat_idx];
    ufbx_texture *tex = get_pbr_texture(mat, map_type);
    if (!tex) return 0;

    /* Prefer relative path, fall back to absolute */
    ufbx_string *path = &tex->relative_filename;
    if (path->length == 0) path = &tex->filename;
    if (path->length == 0) return 0;

    size_t len = path->length;
    if ((int)len >= max_len) len = (size_t)(max_len - 1);
    memcpy(out_path, path->data, len);
    out_path[len] = '\0';

    return (int)len;
}
