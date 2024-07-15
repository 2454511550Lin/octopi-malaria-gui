# can run this to download all the images from a single patient
mkdir -p data/pat
mkdir cropped_images

gsutil -m cp -r \
"gs://octopi-malaria-uganda-2022-data/PAT-071-3_2023-01-22_15-47-3.096602/0/*left_half.bmp" \
"gs://octopi-malaria-uganda-2022-data/PAT-071-3_2023-01-22_15-47-3.096602/0/*right_half.bmp" \
"gs://octopi-malaria-uganda-2022-data/PAT-071-3_2023-01-22_15-47-3.096602/0/*Fluorescence_405_nm_Ex.bmp" \
data/pat