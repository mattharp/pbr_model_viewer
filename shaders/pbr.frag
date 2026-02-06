#version 330 core

// Inputs from vertex shader
in vec3 FragPos;
in vec2 TexCoord;
in mat3 TBN;

// Output
out vec4 FragColor;

// Material textures
uniform sampler2D uBaseColorMap;
uniform sampler2D uMetallicRoughnessMap;
uniform sampler2D uNormalMap;
uniform sampler2D uOcclusionMap;
uniform sampler2D uEmissiveMap;

// Material properties (if no texture)
uniform vec3 uBaseColorFactor;
uniform float uMetallicFactor;
uniform float uRoughnessFactor;
uniform vec3 uEmissiveFactor;

// Texture availability flags
uniform bool uHasBaseColorMap;
uniform bool uHasMetallicRoughnessMap;
uniform bool uHasNormalMap;
uniform bool uHasOcclusionMap;
uniform bool uHasEmissiveMap;

// Lighting
uniform vec3 uLightPositions[3];  // Support 3 lights
uniform vec3 uLightColors[3];
uniform float uLightIntensities[3];
uniform vec3 uCameraPos;
uniform float uAmbientStrength;  // Adjustable ambient lighting strength
uniform vec3 uAmbientColor;  // Ambient light color

// HDR Environment Map (IBL)
uniform samplerCube uEnvMap;
uniform bool uHasEnvMap;
uniform float uEnvIntensity;  // HDR environment brightness

// Material property multipliers (for real-time tweaking)
uniform float uMetallicMultiplier;
uniform float uRoughnessMultiplier;
uniform float uNormalStrength;
uniform float uAOStrength;
uniform float uEmissiveIntensity;
uniform float uEnvRotation;  // HDR rotation in radians

const float PI = 3.14159265359;

// Rotate a direction vector around Y axis
vec3 rotateY(vec3 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

// ============================================================================
// PBR Functions
// ============================================================================

// Normal Distribution Function (GGX/Trowbridge-Reitz)
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return nom / denom;
}

// Geometry Function (Smith's Schlick-GGX)
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// Fresnel Equation (Schlick approximation)
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Fresnel with roughness (for IBL)
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ============================================================================
// Main
// ============================================================================

void main()
{
    // Sample textures
    vec3 albedo = uHasBaseColorMap ? 
        texture(uBaseColorMap, TexCoord).rgb : uBaseColorFactor;
    
    float metallic = uMetallicFactor;
    float roughness = uRoughnessFactor;
    if (uHasMetallicRoughnessMap) {
        vec2 mr = texture(uMetallicRoughnessMap, TexCoord).bg;
        metallic *= mr.y;  // Blue channel
        roughness *= mr.x; // Green channel
    }
    
    // Apply material multipliers for real-time adjustment
    metallic = clamp(metallic * uMetallicMultiplier, 0.0, 1.0);
    roughness = clamp(roughness * uRoughnessMultiplier, 0.04, 1.0);  // Min 0.04 to avoid division by zero
    
    float ao = uHasOcclusionMap ? 
        texture(uOcclusionMap, TexCoord).r : 1.0;
    // Apply AO strength (lerp between full brightness and AO)
    ao = mix(1.0, ao, uAOStrength);
    
    vec3 emissive = uHasEmissiveMap ? 
        texture(uEmissiveMap, TexCoord).rgb * uEmissiveFactor : 
        uEmissiveFactor;
    emissive *= uEmissiveIntensity;
    
    // Get normal from normal map
    vec3 N;
    if (uHasNormalMap) {
        // Sample normal map (in tangent space)
        vec3 tangentNormal = texture(uNormalMap, TexCoord).rgb * 2.0 - 1.0;
        // Apply normal strength (lerp between flat normal and sampled normal)
        tangentNormal.xy *= uNormalStrength;
        tangentNormal = normalize(tangentNormal);
        // Transform to world space using TBN matrix
        N = normalize(TBN * tangentNormal);
    } else {
        // Use interpolated normal
        N = normalize(TBN[2]);  // TBN[2] is the normal
    }
    
    vec3 V = normalize(uCameraPos - FragPos);
    
    // Calculate reflectance at normal incidence
    vec3 F0 = vec3(0.04);  // Base reflectivity for dielectrics
    F0 = mix(F0, albedo, metallic);  // Metals use albedo as F0
    
    // Reflectance equation
    vec3 Lo = vec3(0.0);
    
    for(int i = 0; i < 3; ++i) 
    {
        // Calculate per-light radiance
        vec3 L = normalize(uLightPositions[i] - FragPos);
        vec3 H = normalize(V + L);
        float distance = length(uLightPositions[i] - FragPos);
        // Softer attenuation: constant + linear + quadratic
        float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
        vec3 radiance = uLightColors[i] * uLightIntensities[i] * attenuation;
        
        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;
        
        // Energy conservation
        vec3 kS = F;  // Specular contribution
        vec3 kD = vec3(1.0) - kS;  // Diffuse contribution
        kD *= 1.0 - metallic;  // Metallic surfaces don't have diffuse
        
        // Add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    // Image-Based Lighting (IBL) from HDR environment map
    vec3 iblContribution = vec3(0.0);
    if (uHasEnvMap) {
        vec3 R = reflect(-V, N);
        R = rotateY(R, uEnvRotation);
        
        // Simplified IBL: Sample environment map based on roughness
        // Rougher surfaces = blurrier reflections (approximated with mip level)
        float lod = roughness * 7.0;  // Assume 8 mip levels (0-7)
        vec3 envColor = textureLod(uEnvMap, R, lod).rgb;
        
        // Fresnel for environment
        vec3 F0 = vec3(0.04);
        F0 = mix(F0, albedo, metallic);
        vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
        
        // Specular IBL
        vec3 kS = F;
        vec3 specularIBL = envColor * kS;
        
        // Diffuse IBL (irradiance) - simplified, just sample environment
        vec3 rotatedN = rotateY(N, uEnvRotation);
        vec3 irradiance = texture(uEnvMap, rotatedN).rgb;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        vec3 diffuseIBL = irradiance * albedo * kD;
        
        iblContribution = (diffuseIBL + specularIBL) * uEnvIntensity * ao;
    }
    
    // Ambient lighting (user-adjustable) - used when no HDR or as supplement
    vec3 ambient = uAmbientColor * uAmbientStrength * albedo * ao;
    
    // Final color: ambient + directional lights + IBL + emissive
    vec3 color = ambient + Lo + iblContribution + emissive;
    
    // HDR tonemapping (simple Reinhard)
    color = color / (color + vec3(1.0));
    
    // Subtle contrast enhancement to preserve detail
    color = pow(color, vec3(0.95));
    
    // Gamma correction
    color = pow(color, vec3(1.0/2.2));
    
    FragColor = vec4(color, 1.0);
}
