#here the tiffs are iterated over and stored in different directories as png and jpeg format
from tqdm import tqdm
import SimpleITK as stk



for t in tqdm(range(194)):
  t3 = f"{t:03}"
  feature_path="/Users/apple/Desktop/Fluo-N3DH-CE-2/t"+str(t3)+".tif"
  seg_map_path="/content/Fluo-N3DH-CE-2/01_ST/SEG/man_seg"+str(t3)+".tif"
  
  
  for i in range(len(features_arr)):
    cv2.imwrite('/content/features_jpeg/F'+str(t)+'_'+str(i)+'.jpeg', features_arr[i])
    cv2.imwrite('/content/segmentation_maps_jpeg/L'+str(t)+'_'+str(i)+'.jpeg', seg_arr[i])
    