'''
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = "{:,.4f}".format
'''

'''
df_1w = pd.DataFrame(data=w_matrix)
df_1c = pd.DataFrame(data=C_matrix)
df_1s = pd.DataFrame(data=S_matrix)

#display(df_1w)
#display(df_1c)
#display(df_1s)
'''

'''
def compare_image(img1, img2, title_img1="", title_img2="", figsize=(12,8)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,10))

    #Filtered image
    ax1.set_title(title_img1)
    ax1.tick_params(axis="both",which='both',bottom=False, left=False, labelbottom=False, labelleft=False) 
    ax1.imshow(img1, cmap='gray')

    #Original image
    ax2.set_title(title_img2)
    ax2.tick_params(axis="both",which='both',bottom=False, left=False, labelbottom=False, labelleft=False) 
    ax2.imshow(img2, cmap='gray')

    plt.show()
'''

'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,10))
#Original image
ax1.set_title("Orgiginal img")
ax1.tick_params(axis="both",which='both',bottom=False, left=False, labelbottom=False, labelleft=False) 
ax1.imshow(gray_img, cmap='gray')
#Filtered image
ax2.set_title("Filtered img")
ax2.tick_params(axis="both",which='both',bottom=False, left=False, labelbottom=False, labelleft=False) 
ax2.imshow(f_img, cmap='gray')
plt.show()


fig, axs = plt.subplots(len(sigma_r_list), len(sigma_d_list),figsize=(20,20))
save_dir = r"C:\Users\carot\Downloads\DISIP\pictures\lena_noise\\"

for i, sigma_d in enumerate(sigma_d_list):
    for j, sigma_r in enumerate(sigma_r_list):
        f_img = load_image(os.path.join(save_dir, 'img_'+str(sigma_d)+'_'+str(sigma_r)+'.jpg'))
        
        axs[j,i].set_title("σ_d={} σ_r={}".format(sigma_d, sigma_r))
        axs[j,i].tick_params(axis="both",which='both',bottom=False, left=False, labelbottom=False, labelleft=False) 
        axs[j,i].imshow(f_img, cmap='gray')
'''


'''
save_dir = r"C:\Users\carot\Downloads\DISIP\pictures\lena_noise_color"
image_path = r'C:\Users\carot\Downloads\DISIP\pictures\lena_noise_color.jpg'

rgb_img, lab_img = load_image(image_path, False)
sigma_d_list = [0.5,1,5,20,80,150]
sigma_r_list = [1,10,50,100,300,500]

padding = 0
window_size=43
img_collage = np.zeros((rgb_img.shape[0]*len(sigma_r_list) + padding*(len(sigma_r_list)+1),
                       rgb_img.shape[1]*len(sigma_d_list) + padding*(len(sigma_d_list)+1),3))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for i, sigma_d in enumerate(sigma_d_list):
    print("\n[{}/{}]:{}".format(i+1, len(sigma_d_list),sigma_d), end="\n\t")
    for j, sigma_r in enumerate(sigma_r_list):
        print("[{}/{}]:{}".format(j+1, len(sigma_r_list),sigma_r), end='  |  ')

        f_img_lab = bilateral_filter(lab_img, sigma_d, sigma_r, window_size)
        f_img_rgb = (color.lab2rgb(f_img_lab)*255).astype(np.uint8)
        
        #Save the filtered image
        cv2.imwrite(os.path.join(save_dir, 'img_'+str(sigma_d)+'_'+str(sigma_r)+'.jpg'), cv2.cvtColor(f_img_rgb, cv2.COLOR_BGR2RGB))    
        img_collage[j*rgb_img.shape[0]+j*padding:(j+1)*rgb_img.shape[0]+j*padding,
                    i*rgb_img.shape[1]+i*padding:(i+1)*rgb_img.shape[1]+i*padding,:] = cv2.cvtColor(f_img_rgb, cv2.COLOR_BGR2RGB)

cv2.imwrite(os.path.join(save_dir, 'collage.jpg'), img_collage)
'''

'''
Put the iteration number on the image and create a video with the sequence of the iterations

from PIL import Image, ImageFont, ImageDraw, ImageOps
import glob
dir_path = r".\pictures\paine_it"
save_path = os.path.join(dir_path, 'label')
if not os.path.exists(save_path):
    os.mkdir(save_path)

frames = []
b_border, w_border = 5, 50
for i,filename in enumerate(glob.glob(dir_path+"\*.jpg")):
    img = Image.open(filename)
    img = ImageOps.expand(img, border=(0, w_border, 0, 0), fill='white')
    img = ImageOps.expand(img, border=(b_border, b_border, b_border, b_border), fill='black')
    W, H = img.size
    
    text = "Iteration "+str(i+1)
    
    im = Image.new("RGBA",(W,H),"yellow")
    draw = ImageDraw.Draw(im)
    w, h = draw.textsize(text)
    
    font = ImageFont.truetype("C:\\Windows\\Fonts\\Georgia.ttf", 36)
    image_editable = ImageDraw.Draw(img)
    image_editable.text(((W-w)/2,8), text, (0, 0, 0), font=font)
    
    frames.append(img)
    img.save(os.path.join(save_path, "it_{:03d}.jpg").format(i))

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('.\\pictures\\output_video.mp4',fourcc, 20.0, (W, H))
for filename in glob.glob(save_path+'//*.jpg'):
    img = cv2.imread(filename)
    for _ in range(20):
        out.write(img)
out.release()
'''