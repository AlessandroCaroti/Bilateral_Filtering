

'''
df_1w = pd.DataFrame(data=w_matrix)
df_1c = pd.DataFrame(data=C_matrix)
df_1s = pd.DataFrame(data=S_matrix)

#display(df_1w)
#display(df_1c)
#display(df_1s)




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