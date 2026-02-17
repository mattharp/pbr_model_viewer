/*
 * ufbx_bridge.h
 * Thin C bridge over ufbx for Python ctypes consumption.
 * Flattens ufbx's complex nested structs into simple arrays.
 *
 * Compile together with ufbx.c:
 *   gcc -shared -O2 -o ufbx.dll ufbx.c ufbx_bridge.c -DUFBX_BRIDGE_EXPORT -lm
 */

#ifndef UFBX_BRIDGE_H
#define UFBX_BRIDGE_H

#include <stdint.h>

#ifdef UFBX_BRIDGE_EXPORT
    #ifdef _WIN32
        #define BRIDGE_API __declspec(dllexport)
    #else
        #define BRIDGE_API __attribute__((visibility("default")))
    #endif
#else
    #define BRIDGE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a loaded FBX scene */
typedef void* fbx_scene_handle;

/* Map type identifiers for texture queries */
enum {
    FBX_MAP_BASE_COLOR = 0,
    FBX_MAP_METALLIC   = 1,
    FBX_MAP_ROUGHNESS  = 2,
    FBX_MAP_NORMAL     = 3,
    FBX_MAP_EMISSIVE   = 4,
    FBX_MAP_OCCLUSION  = 5,
    FBX_MAP_COUNT      = 6
};

/* PBR material properties (flat struct for ctypes) */
typedef struct {
    float base_color[4];   /* RGBA */
    float metallic;
    float roughness;
    float emissive[3];     /* RGB */
    int   has_texture[FBX_MAP_COUNT]; /* 1 if map type has a texture */
} fbx_material_info;

/* Embedded texture data */
typedef struct {
    const void *data;      /* Raw image bytes (PNG/JPEG/etc) */
    uint32_t    size;      /* Size in bytes */
    char        filename[512]; /* Original filename hint */
} fbx_texture_info;

/* ---- Scene loading / freeing ---- */

BRIDGE_API fbx_scene_handle fbx_load(const char *filepath);
BRIDGE_API void             fbx_free(fbx_scene_handle scene);

/* ---- Scene queries ---- */

BRIDGE_API int fbx_mesh_count(fbx_scene_handle scene);
BRIDGE_API int fbx_material_count(fbx_scene_handle scene);

/* ---- Mesh data extraction ---- */

/*
 * Get total triangulated vertex and index counts for a mesh.
 * Call this first to allocate buffers.
 */
BRIDGE_API void fbx_mesh_get_counts(fbx_scene_handle scene, int mesh_idx,
                                    int *out_vertex_count, int *out_index_count);

/*
 * Extract triangulated mesh data into pre-allocated flat arrays.
 * Each vertex has: position(3) + normal(3) + uv(2) + tangent(3) = 11 floats.
 * Indices are uint32.
 * Returns actual unique vertex count after deduplication.
 */
BRIDGE_API int fbx_mesh_extract(fbx_scene_handle scene, int mesh_idx,
                                float *out_vertices, uint32_t *out_indices,
                                int max_vertices, int max_indices);

/*
 * Get material index assigned to a mesh (first material).
 * Returns -1 if no material.
 */
BRIDGE_API int fbx_mesh_material_index(fbx_scene_handle scene, int mesh_idx);

/* ---- Material queries ---- */

BRIDGE_API void fbx_material_get_info(fbx_scene_handle scene, int mat_idx,
                                      fbx_material_info *out_info);

/*
 * Get embedded texture data for a material's map type.
 * Returns 1 on success, 0 if no embedded data.
 */
BRIDGE_API int fbx_material_get_texture(fbx_scene_handle scene, int mat_idx,
                                        int map_type, fbx_texture_info *out_info);

/*
 * Get the file path of an external (non-embedded) texture.
 * Returns string length, 0 if none.
 */
BRIDGE_API int fbx_material_get_texture_path(fbx_scene_handle scene, int mat_idx,
                                             int map_type, char *out_path, int max_len);

#ifdef __cplusplus
}
#endif

#endif /* UFBX_BRIDGE_H */
