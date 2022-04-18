from utranslate.setup import my_setup
import os
import time

def do_the_setup():
    print("\ndownload size : \n\ttrained word embeddings ~ 2.4GB \n\tdataset ~ 100MB \n\n.... this might take some time depending on your internet speed…\n\n")

    setup = my_setup()

    start = time.time()
    print('█ ',end='')
    #downloading the files...
    setup.download_files()
    print('█ █ █ █ ',end='')


    #extracting the files...
    setup.extract_files()
    print('█ █ █ ',end='')

    #populating all the data files...
    setup.populate_data_files()
    print('█ █ ',end='')

    #cleaning up...
    setup.clean_up()
    print('█ █',end='')

    print('\n\nDownload completed!')

    print(f"\ntotal time - > {int((time.time()-start)//60)} min and {int((time.time()-start)%60)} sec")


build_data_path = os.environ.get('utranlate_build_data_path')

if build_data_path == None:
    print("Welcome to utarnslate build")
    print("please read the utranslate build guide for complete information\n\n")
    print("Do you want to download the default dataset and embeddings?")
    ans = input()

    if ans.lower() == 'y' or ans.lower() == 'yes':
        do_the_setup()
    else:
        print("You will not be able to train and use the inbuild model!!\n ")
        i = 0
        while(i < 10):
            print('█ ',end='')
            time.sleep(0.5)
            i+=1

    print("\n setup completed .. ")
