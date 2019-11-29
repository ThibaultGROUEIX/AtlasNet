# Training

:raised_hand_with_fingers_splayed: Monitor your training on http://localhost:8888/

![visdom](/Users/mac-mud/Projects_tibo/AtlasNet/doc/pictures/visdom2.png)



##Quantitative Results 


The number reported are the chamfer distance, the f-score and the [metro](https://github.com/ThibaultGROUEIX/AtlasNet/issues/34) distance.

| Method                 | Chamfer⁽⁰⁾ | Fscore | Metro | GPU memory | Total Train time |
| ---------------------- | ---------- | ------ | ----- | ---------- | ---------------- |
| Autoencoder 25 Squares | -          | -      | -     | -          | -                |
| Autoencoder 1 Sphere   | -          | -      | -     | -          | -                |
| SingleView 25  Squares | -          | -      | -     | -          | -                |
| SingleView 1 Sphere    | -          | -      | -     | -          | -                |

⁽⁰⁾  computed between 2500 ground truth points and 2500 reconstructed points.

⁽¹⁾ with the flag ```--accelerated_chamfer 1```.

⁽²⁾this is only an estimate, the code is not optimised.  The easiest way to enhance it would be to preload the training data to use the GPU at 100%. Time computed with the flag ```--accelerated_chamfer 1```.
Visualisation 





# Paper reproduction 

To reproduce main results from the paper : ```python ./training/launch.py --mode train```

In case you need the results of ICP on PointSetGen output :

* [ICP on PSG](https://cloud.enpc.fr/s/3a7Xg9RzIsgmofw)

