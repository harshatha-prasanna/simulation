# How a MuJoCo simulation goes from file to picture

```
XML file  →  MuJoCo compiles it into a "model"  →  physics runs  →  renderer draws a picture
(describe)    (Python asks: MjModel.from_xml_path)   (mj_step)        (Renderer)
```

1. **XML file (describe):** `assets/descriptions/DropCubeInBinEnv.xml` describes the world: objects, shapes, positions, joints and the home pose. Nothing runs here.
2. **Compile (`mj.MjModel.from_xml_path`):** MuJoCo, a C physics engine driven from Python, reads the XML and builds a model.
3. **Physics (`mj.mj_step`):** each step moves time forward and works out where everything is from the joint angles, gravity and contacts.
4. **Render (`mj.Renderer`):** draws the scene from a camera. How things **look** comes from the asset files (meshes, textures, colors); where things **are** comes from the current physics state.
