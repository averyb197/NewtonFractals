# NewtonFractals
Makes newton fractals in the complex plane. I suggest playing with the playground notebook in google colab, and set the runtime to a GPU so you can compute massive fractals quickly. Don't set the resolution over 10000 and if you want high quality images make sure the dpi is set accordingly when saving images otherwise you are doing way too much for nothing. 

If you have a GPU, I'd reccomend using main.py which will output a .npy file which you can load into an image, template for how to do this is in the notebook. 

plt.imshow will not show the full resolution, but there is a way to make it zoomable, but I was too lazy to add this, also for really high reolution it will violate you pc. 



