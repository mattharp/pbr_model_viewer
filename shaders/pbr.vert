#version 330 core

// Vertex attributes
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec3 aTangent;

// Outputs to fragment shader
out vec3 FragPos;
out vec2 TexCoord;
out mat3 TBN;  // Tangent-Bitangent-Normal matrix for normal mapping

// Uniforms
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;  // transpose(inverse(mat3(model)))

void main()
{
    // Transform position to world space
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    FragPos = worldPos.xyz;
    
    // Pass through texture coordinates
    TexCoord = aTexCoord;
    
    // Calculate TBN matrix for normal mapping
    vec3 T = normalize(uNormalMatrix * aTangent);
    vec3 N = normalize(uNormalMatrix * aNormal);
    
    // Re-orthogonalize T with respect to N (Gram-Schmidt)
    T = normalize(T - dot(T, N) * N);
    
    // Calculate bitangent
    vec3 B = cross(N, T);
    
    // Construct TBN matrix (transforms from tangent space to world space)
    TBN = mat3(T, B, N);
    
    // Final position
    gl_Position = uProjection * uView * worldPos;
}
