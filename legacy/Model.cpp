#ifdef LEGACY_MODEL

#include <iostream>
#include <glm/gtx/transform.hpp>
#include <tinygltf/tiny_gltf.h>

struct Image {
	int32_t component;
	int32_t width;
	int32_t height;
	std::vector<uint8_t> data;
};

struct Buffer {
	glm::mat4 transform;
	std::vector<uint16_t> indices;
	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> texcoords;
	Image image;
};

class Model {
public:
	std::vector<Buffer> buffers;

	Model(std::string filename) {
		std::string error, warning;
		tinygltf::TinyGLTF loader;
		tinygltf::Model model;

		bool result = loader.LoadBinaryFromFile(&model, &error, &warning, filename);

		if (!warning.empty())
			std::cout << "GLTF Warning: " << warning << std::endl;
		if (!error.empty())
			std::cout << "GLTF Error: " << error << std::endl;
		if (!result)
			std::cout << "GLTF model could not be loaded!" << std::endl;

		auto &scene = model.scenes.at(model.defaultScene);

		for (auto &nodeIndex : scene.nodes)
			loadNode(model, model.nodes.at(nodeIndex), glm::mat4{1.0f});
	}

private:
	void loadNode(tinygltf::Model &model, tinygltf::Node &node, glm::mat4 transform) {
		glm::mat4 scale{1.0f}, rotation{1.0f}, translation{1.0f};

		if (!node.rotation.empty())
			rotation = glm::rotate(static_cast<float_t>(node.rotation.at(3)),
								   glm::vec3{node.rotation.at(0), node.rotation.at(1), node.rotation.at(2)});
		for (int i = 0; i < node.scale.size(); i++)
			scale[i][i] = node.scale.at(i);
		for (int i = 0; i < node.translation.size(); i++)
			translation[3][i] = node.translation.at(i);

		transform = transform * translation * rotation * scale;

		if (node.mesh >= 0 && node.mesh < model.meshes.size())
			loadMesh(model, model.meshes.at(node.mesh), transform);

		for (auto &childIndex : node.children)
			loadNode(model, model.nodes.at(childIndex), transform);

		for (auto &material : model.materials) {
			for (auto &value : material.values) {
				if (!value.first.compare("baseColorTexture")) {
					auto &target = buffers.at(value.second.TextureTexCoord()).image;
					auto &source = model.images.at(model.textures.at(value.second.TextureIndex()).source);

					target.component = source.component;
					target.width = source.width;
					target.height = source.height;
					target.data = source.image;
				}
			}
		}
	}

	void loadMesh(tinygltf::Model &model, tinygltf::Mesh &mesh, glm::mat4 transform) {
		for (auto &primitive : mesh.primitives) {
			buffers.push_back(Buffer{});
			auto &buffer = buffers.back();
			buffer.transform = transform;

			auto &indexView = model.bufferViews.at(primitive.indices);
			auto &indexBuffer = model.buffers.at(indexView.buffer);

			buffer.indices.resize(indexView.byteLength / sizeof(uint16_t));
			std::memcpy(buffer.indices.data(),
						indexBuffer.data.data() + indexView.byteOffset, indexView.byteLength);

			for (auto &attribute : primitive.attributes) {
				auto &accessor = model.accessors.at(attribute.second);
				auto &primitiveView = model.bufferViews.at(accessor.bufferView);
				auto &primitiveBuffer = model.buffers.at(primitiveView.buffer);

				if (attribute.first.compare("POSITION") == 0) {
					buffer.positions.resize(primitiveView.byteLength / sizeof(glm::vec3));
					std::memcpy(buffer.positions.data(),
								primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
				} else if (attribute.first.compare("NORMAL") == 0) {
					buffer.normals.resize(primitiveView.byteLength / sizeof(glm::vec3));
					std::memcpy(buffer.normals.data(),
								primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
				} else if (attribute.first.compare("TEXCOORD_0") == 0) {
					buffer.texcoords.resize(primitiveView.byteLength / sizeof(glm::vec2));
					std::memcpy(buffer.texcoords.data(),
								primitiveBuffer.data.data() + primitiveView.byteOffset, primitiveView.byteLength);
				}
			}
		}
	}
};

#endif //LEGACY_MODEL