# Neural Painter

We use a random neural network: f(x, y) -> pixel to generate image.
Under some architectural hypothesis, there are plenty of hyper parameters to be fiddle with.

# Usage
```bash
./neural_painter.py --image_size 800x800 --hidden_size 100 --nr_hidden 4 --nonlin random_every_time --nr_channel 3 --output_nonlin identity --coord_bias --seed 42 --output 42.png
```

# Gallery

<img class="screenshots" src="gallery/3-batch_norm:False-batch_norm_position:before_nonlin-coord_bias:True-hidden_size:100-image_size:1366x768-nonlin:random_every_time-nr_channel:3-nr_hidden:4-output_nonlin:identity-recurrent:False-seed:3-use_bias:False.jpg" alt="example image">


# Related Links

- image regression from karpathy:
	http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html
- high resolution MNIST generator: (CPPN + GAN/VAE)
	http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/
