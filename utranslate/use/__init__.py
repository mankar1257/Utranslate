from utranslate.setup import my_setup
import os
import time

def do_the_setup():

    setup = my_setup()

    start = time.time()

    translators = [0,1]

    print("\nPlease select the translators to download ( to select all hit enter ) \n\n Available translators :\n\t 0 hi_to_en\n \n\t 1 en_to_hi\n \n( enter the space-separated indexes )")
    ans = input()
    if ans == "":
        for t in translators:
            setup.download_predict(t)
    else:
        for val in ans.split(' '):
            if int(val) in translators:
                setup.download_predict(int(val))
            else:
                print("please enter the valid number")

    i = 0
    while(i < 10):
        print('█ ',end='')
        time.sleep(0.5)
        i+=1
        
    print('\n\nDownload completed!')

    print(f"\ntotal time - > {int((time.time()-start)//60)} min and {int((time.time()-start)%60)} sec")


build_data_path = os.environ.get('utranlate_build_data_path')

if build_data_path == None:
    print("Welcome to utarnslate use")
    print("please read the utranslate use guide for complete information\n")
    
    print("\nin order to use the pre-trained translators you need to download the translators\n")

    print("\t Do you want to download the translators? (y/n)")
    ans = input()

    if ans.lower() == 'y' or ans.lower() == 'yes':
        do_the_setup()
        
    else:
        print("You will not be able to use the available translation services!\n ")
        i = 0
        while(i < 10):
            print('█ ',end='')
            time.sleep(0.5)
            i+=1

    print("\n setup completed .. ")

