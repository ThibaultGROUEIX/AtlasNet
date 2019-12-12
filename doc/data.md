# Data Preprocessing



We used shapenetv1 55 categories

[1] Shoot rays from infinity on each models to uniformly sample it. -> obj files

[2] Sample 30k points in each obj. -> nay files

The **renderings** for 13 categories are from *3DR2N2 Choy et al*.



#### Track of some file errors

* FileNotFoundError: [Errno 2] No such file or directory: '/home/thibault/ssd/AtlasNet/dataset/data/ShapeNetV1Renderings/02958343/f9c1d7748c15499c6f2bd1c4e9adb41/rendering/00.png'

  ​      :arrow_right: File removed from the point clouds because it doesn’t exist in the renderings



* 03001627/56262eebe592b085d319c38340319ae4/rendering/07.png CHECK IT. Pierre-Alain had an issue with this file.
* [‘data/shapenetcore/02958343/npy/603269ee27eb7a4c599d9c8430d600ff.points.ply.npy’]