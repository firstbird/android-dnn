#include "Shader.h"

#include "AndroidOut.h"
#include "Model.h"
#include "Utility.h"

#include <cstddef>

Shader::Shader(GLuint program, GLint position, GLint uv,
    GLint modelMatrix,// mvp
    GLint viewMatrix,// mvp
               GLint projectionMatrix)
    : program_(program),
      position_(position),
      uv_(uv),
      modelMatrix_(modelMatrix),// mvp
      viewMatrix_(viewMatrix),// mvp
      projectionMatrix_(projectionMatrix) {}

Shader::~Shader() {
    if (vao_ != 0) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (vbo_ != 0) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (ebo_ != 0) {
        glDeleteBuffers(1, &ebo_);
        ebo_ = 0;
    }
    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }
}

void Shader::initBufferObjects() {
    if (vao_ != 0) {
        return;
    }
    // 【gen】
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);
    // 【bind】
    glBindVertexArray(vao_);//1
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    // 【set】
    glVertexAttribPointer(position_, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));//要在1之后
    glEnableVertexAttribArray(position_);
    // 【set】
    glVertexAttribPointer(uv_, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, uv)));
    glEnableVertexAttribArray(uv_);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);// 解绑vbo
    glBindVertexArray(0);// 解绑vao
}

Shader *Shader::loadShader(
        const std::string &vertexSource,
        const std::string &fragmentSource,
        const std::string &positionAttributeName,
        const std::string &uvAttributeName,
        const std::string &modelMatrixUniformName,// mvp
        const std::string &viewMatrixUniformName,// mvp
        const std::string &projectionMatrixUniformName) {
    Shader *shader = nullptr;

    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, vertexSource);
    if (!vertexShader) {
        return nullptr;
    }

    GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentSource);
    if (!fragmentShader) {
        glDeleteShader(vertexShader);
        return nullptr;
    }

    GLuint program = glCreateProgram();
    if (program) {
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);

        glLinkProgram(program);
        GLint linkStatus = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
        if (linkStatus != GL_TRUE) {
            GLint logLength = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

            // If we fail to link the shader program, log the result for debugging
            if (logLength) {
                GLchar *log = new GLchar[logLength];
                glGetProgramInfoLog(program, logLength, nullptr, log);
                aout << "Failed to link program with:\n" << log << std::endl;
                delete[] log;
            }

            glDeleteProgram(program);
        } else {
            // Get the attribute and uniform locations by name. You may also choose to hardcode
            // indices with layout= in your shader, but it is not done in this sample
            GLint positionAttribute = glGetAttribLocation(program, positionAttributeName.c_str());
            GLint uvAttribute = glGetAttribLocation(program, uvAttributeName.c_str());
            GLint modelMatrixUniform = glGetUniformLocation(program, modelMatrixUniformName.c_str());// mvp
            GLint viewMatrixUniform = glGetUniformLocation(program, viewMatrixUniformName.c_str());// mvp
            GLint projectionMatrixUniform = glGetUniformLocation(
                    program,
                    projectionMatrixUniformName.c_str());

            // Only create a new shader if all the attributes are found.
            if (positionAttribute != -1
                && uvAttribute != -1
                && modelMatrixUniform != -1
                && viewMatrixUniform != -1
                && projectionMatrixUniform != -1) {

                shader = new Shader(
                        program,
                        positionAttribute,
                        uvAttribute,
                        modelMatrixUniform,
                        viewMatrixUniform,
                        projectionMatrixUniform);
                shader->initBufferObjects();
            } else {
                glDeleteProgram(program);
            }
        }
    }

    // The shaders are no longer needed once the program is linked. Release their memory.
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shader;
}

GLuint Shader::loadShader(GLenum shaderType, const std::string &shaderSource) {
    Utility::assertGlError();
    GLuint shader = glCreateShader(shaderType);
    if (shader) {
        auto *shaderRawString = (GLchar *) shaderSource.c_str();
        GLint shaderLength = shaderSource.length();
        glShaderSource(shader, 1, &shaderRawString, &shaderLength);
        glCompileShader(shader);

        GLint shaderCompiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &shaderCompiled);

        // If the shader doesn't compile, log the result to the terminal for debugging
        if (!shaderCompiled) {
            GLint infoLength = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLength);

            if (infoLength) {
                auto *infoLog = new GLchar[infoLength];
                glGetShaderInfoLog(shader, infoLength, nullptr, infoLog);
                aout << "Failed to compile with:\n" << infoLog << std::endl;
                delete[] infoLog;
            }

            glDeleteShader(shader);
            shader = 0;
        }
    }
    return shader;
}

void Shader::activate() const {
    glUseProgram(program_);
}

void Shader::deactivate() const {
    glUseProgram(0);
}

void Shader::drawModel(const Model &model) const {
    /*
    // 旧实现：客户端内存指针（GLES3 规范不推荐；顶点留在 CPU 堆上）
    glVertexAttribPointer(position_, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), model.getVertexData());
    glEnableVertexAttribArray(position_);
    glVertexAttribPointer(uv_, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          ((uint8_t *) model.getVertexData()) + sizeof(Vector3));
    glEnableVertexAttribArray(uv_);
    if (model.getTexturePtr()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, model.getTexture().getTextureID());
    }
    glDrawElements(GL_TRIANGLES, model.getIndexCount(), GL_UNSIGNED_SHORT, model.getIndexData());
    glDisableVertexAttribArray(uv_);
    glDisableVertexAttribArray(position_);
    */

    if (model.getVertexCount() == 0 || model.getIndexCount() == 0) {
        return;
    }

    //glVertexAttribPointer和glEnableVertexAttribArray在initBufferObjects中已经设置过了，这里只需要绑定vbo和ebo
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(sizeof(Vertex) * model.getVertexCount()),
                 model.getVertexData(),
                 GL_STREAM_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(sizeof(Index) * model.getIndexCount()),
                 model.getIndexData(),
                 GL_STREAM_DRAW);

    if (model.getTexturePtr()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, model.getTexture().getTextureID());
    }

    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(model.getIndexCount()), GL_UNSIGNED_SHORT,
                   reinterpret_cast<void*>(0));

    // 先解绑 VAO，避免在 VAO 仍绑定期间把 GL_ELEMENT_ARRAY_BUFFER 绑成 0（会清掉 VAO 内保存的 EBO）
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Shader::setModelMatrix(const float *modelMatrix) const {
    glUniformMatrix4fv(modelMatrix_, 1, GL_FALSE, modelMatrix);
}// mvp

void Shader::setViewMatrix(const float *viewMatrix) const {
    glUniformMatrix4fv(viewMatrix_, 1, GL_FALSE, viewMatrix);
}// mvp

void Shader::setProjectionMatrix(const float *projectionMatrix) const {
    glUniformMatrix4fv(projectionMatrix_, 1, GL_FALSE, projectionMatrix);
}// mvp