The idea is to use previous generated images to create new ones, using their generated latents. 
This works as a good ControlNet variation, but also allows to create completely new compositions, additional benefit i observed: using same generated info with same prompt might provide you with an excellent result in definition and in the final composition as the model will focus faster on the right elements....

This works (for only one latent or many) saving first their generated latents as numpy files and then applying them during first steps of a new generation.
Working with one latent it's easy, but using many might be tricky.

Schedulers: Only Euler Ancestral and DDIM (with ETA=1), good results could start to be observed from 16-20 steps, multiplier:0'4 to 0'55 and strengh:0'8 - 0'9, guidance: 8-14

-Checkbox: "Save generated latents": save the latent for the current on-going generation (or while it is activated) into the latents subdir, to re-use them on further generations
-Checkbox: "Load latent from generation" load the generated latent and apply it to the next generation
-Name of Numpy latens: the latents are saved as .npy or numpy files, here you introducte the name (or names) of the numpy files to be applied (more on this later).
-Formula, when using more than one latent you will need to define here how they will create the final image (more on this later)..

IMPORTANT: you are only allowed to use latents (or sumatory of them) of same size to the output image. If you are using a latent for a generated 512x512 img you will be only allowed to use it for a new 512x512 image, or ir you use two latents,  you will be able to sum them alongside the dimension that is equal in both files, adding the total size of the other dimension:one is 384x640 latent and another 256x640, result 640x640), trying to create a composition with complete diferent sizes will not work. ( cannot add 256x512 to 384x640....), be careful with this. Try first with only one file and get used to how it works prior trying to work with many latents....
Example: one latent (width X height) 256x704 is added horizontally to a vertical composition of two other latents 384x192 and 384x512 (sum=384x640) : 640x704.


Name of Numpy- Latent:
You must write down here all the names of the latents files (.npy) you will use, comma separated, you need to assign them an index to be use a reference at the formula area.
Format is: "index":"name of numpy file with extension.npy", "index2":"name of same or different numpy file with extension.npy", etc...(the max index allowed needs to be the equal to the total of files to name, if not it will not work)


The formula indicating the order and the direction for the sumatory of the files, use parentesis marks for enclosing operations according to their relevancy
For the previous example:one latent (width X height) 256x704 is added horizontally to a vertical composition of two other latents 384x192 and 384x512 (sum=384x640) : 640x704.
This will be need to be write down this way:

Files: "1:file1.npy,2:file2.npy,3:another.npy"
Formula: 1w(2h3)
w: witdh, horizontal sumatory (remember, their height must be the same)
h: height, vertical sumatory (remember, their width must be the same)
This is a horizontal sumatory of 1 with the vertical sumatory of 2 and 3 :
Yoy might complicate it as much as you want, as the dimension of the files is ok and the total size obtained is the same as the proposed generation size.
i.e.  1h((2w4)h3) 1 vertical(height) sum of the result of ((2 horizonal sum with 4) and vertical sum of 3)
Formula for one file : only provided index of the file ("1:file1.npy,2:file2.npy,3:another.npy" index are 1,2 or 3)
