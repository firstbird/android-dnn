#ifndef ANDROIDGLINVESTIGATIONS_SHADER_H
#define ANDROIDGLINVESTIGATIONS_SHADER_H

#include <string>
#include <GLES3/gl3.h>

class Model;

/*!
 * A class representing a simple shader program. It consists of vertex and fragment components. The
 * input attributes are a position (as a Vector3) and a uv (as a Vector2). It also takes a uniform
 * uniforms uModel, uView, uProjection（列主序，gl_Position = P * V * M * 顶点）。The shader expects a single texture for
 * fragment shading, and does no other lighting calculations (thus no uniforms for lights or normal
 * attributes).
 */
class Shader {
public:
    /*!
     * Loads a shader given the full sourcecode and names for necessary attributes and uniforms to
     * link to. Returns a valid shader on success or null on failure. Shader resources are
     * automatically cleaned up on destruction.
     *
     * @param vertexSource The full source code for your vertex program
     * @param fragmentSource The full source code of your fragment program
     * @param positionAttributeName The name of the position attribute in your vertex program
     * @param uvAttributeName The name of the uv coordinate attribute in your vertex program
     * @param modelMatrixUniformName uModel（物体局部→世界）
     * @param viewMatrixUniformName uView（世界→相机）
     * @param projectionMatrixUniformName uProjection（相机→裁剪，本工程 2D 下多为单位阵）
     * @return a valid Shader on success, otherwise null.
     */
    static Shader *loadShader(
            const std::string &vertexSource,
            const std::string &fragmentSource,
            const std::string &positionAttributeName,
            const std::string &uvAttributeName,
            const std::string &modelMatrixUniformName,// mvp
            const std::string &viewMatrixUniformName,// mvp
            const std::string &projectionMatrixUniformName);

    ~Shader();

    /*!
     * Prepares the shader for use, call this before executing any draw commands
     */
    void activate() const;

    /*!
     * Returns the GL program ID
     */
    [[nodiscard]] inline GLuint getProgram() const { return program_; }

    /*!
     * Cleans up the shader after use, call this after executing any draw commands
     */
    void deactivate() const;

    /*!
     * Renders a single model
     * @param model a model to render
     */
    void drawModel(const Model &model) const;

    void setModelMatrix(const float *modelMatrix) const;// mvp
    void setViewMatrix(const float *viewMatrix) const;// mvp
    void setProjectionMatrix(const float *projectionMatrix) const;// mvp

private:
    /*!
     * Helper function to load a shader of a given type
     * @param shaderType The OpenGL shader type. Should either be GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
     * @param shaderSource The full source of the shader
     * @return the id of the shader, as returned by glCreateShader, or 0 in the case of an error
     */
    static GLuint loadShader(GLenum shaderType, const std::string &shaderSource);

    /*!
     * Constructs a new instance of a shader. Use @a loadShader
     * @param program the GL program id of the shader
     * @param position the attribute location of the position
     * @param uv the attribute location of the uv coordinates
     */
    Shader(
            GLuint program,
            GLint position,
            GLint uv,
            GLint modelMatrix,
            GLint viewMatrix,
            GLint projectionMatrix);

    /** 创建 VAO/VBO/EBO 并配置顶点属性（须在 GL 上下文 current 时调用，由 loadShader 内部调用） */
    void initBufferObjects();

    GLuint program_;
    GLint position_;
    GLint uv_;
    GLint modelMatrix_;// mvp
    GLint viewMatrix_;// mvp
    GLint projectionMatrix_;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLuint ebo_ = 0;
};

#endif //ANDROIDGLINVESTIGATIONS_SHADER_H