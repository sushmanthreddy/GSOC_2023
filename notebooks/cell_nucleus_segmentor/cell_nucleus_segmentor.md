# Cell Nucleus Segmentor 

Here in this model , we are fine-tuning the segment anything model from [MetaAI](https://segment-anything.com)

## Dataset Description

*  The dataset we are using is from cell tracking challenge dataset , which has            segmentation maps and flouroscene images and it also has centroid position of the 
sample flouscene image 
### Flourscence image 
<img src="/assets/cellsegmentor_sam/Screenshot 2023-06-06 at 12.09.19 AM.png" width=300/>

* tT.tif - Multi-page tiff file that contains the original image data (i.e., either 8-bit or 16-bit image data depending on the dataset) of a given frame.

### Segmentation maps

<img src="/assets/cellsegmentor_sam/Screenshot 2023-06-06 at 10.14.11 AM.png" width=300/>

* man_segT.tif - 16-bit multi-page tiff file (segmented objects have unique positive labels that are not necessarily propagated over time, background has zero label). It contains reference segmentation for the corresponding original image tT.tif. In the case of gold segmentation truth, only selected frames are annotated (i.e., the man_segT.tif file does not have to be provided for every tT.tif file). However, in those frames, all objects are segmented. In the case of silver segmentation truth, all frames tend to be completely annotated. Nevertheless, because of being difficult to segment automatically, some objects may be missing there. If that involves all objects in a particular frame, the reference segmentation annotation is not released at all.

### Ground truth markers

<img src="/assets/cellsegmentor_sam/Screenshot 2023-06-06 at 10.16.12 AM.png" width=300/>

* man_trackT.tif - 16-bit multi-page tiff file (markers have unique positive labels propagated over time, background has zero label). It contains markers for the corresponding original image tT.tif. The man_trackT.tif file is provided for every challenge tT.tif file. However, note that the man_trackT.tif file does not have to be provided for every training tT.tif file. Only the frames with non-empty reference tracking annotation are released.



