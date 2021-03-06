# CycleGAN


CycleGAN by Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros The CycleGAN is an extension of the GAN architecture that involves the simultaneous training of two generator models and two discriminator models. One generator takes images from the first domain as input and outputs images for the second domain during the training, and the other generator takes images from the second domain as input and generates images for the first domain. Discriminator models are then used to determine how plausible the generated images are and outputs a decision variable, whether it is real or fake. This extension alone might be enough to generate plausible images in each domain, but not sufficient to generate translations of the input images.The CycleGAN uses an additional extension to the architecture called cycle consistency. This is the idea that an image output by the first generator could be used as input to the second generator and the output of the second generator should match the original image.[1] The reverse is also true: that an output from the second generator can be fed as input to the first generator and the result should match the input to the second generator.Cycle consistency is a concept from machine translation where a phrase translated from Turkish to English should translate from English back to Turkish and be identical to the original phrase. The reverse process should also be true. The CycleGAN encourages cycle consistency by adding an additional loss to measure the difference between the generated output of the second generator and the original image, and the reverse. This acts as a regularization of the generator models, guiding the image generation process in the new domain toward image translation.


![alt text](https://lh3.googleusercontent.com/-BPFGgfI_F9A/YAm4tdOAxYI/AAAAAAAACZA/AxR1Uanez-A0Ckl1oYTuOdlqodGh0zOywCLcBGAsYHQ/w320-h177/image.png)


![alt text](https://lh3.googleusercontent.com/-1Ym-YTiyBS8/YAm5GE6jLnI/AAAAAAAACZU/uP6q6BCgyNIq5x_Y8aCrhpmNdZU5Pr8owCLcBGAsYHQ/w615-h321/image.png)


![alt text](https://lh3.googleusercontent.com/-v6vQ6F4enyo/YAm5i2pJPlI/AAAAAAAACZk/T6SXjrnmGesCkxAVNq1v9AA3xUmYySWjQCLcBGAsYHQ/image.png)
