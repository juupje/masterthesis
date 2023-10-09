global_choice = None
def prompt_overwrite(name:str):
    global global_choice
    if(global_choice):
        overwrite = global_choice.lower()
    else:
        while True:
            overwrite = input(f"{name} already exists, do you want to overwrite[y/n,Y/N=all]: ")
            if(overwrite in "YN"):
                global_choice = overwrite
                overwrite = overwrite.lower()
                break
            elif(overwrite in "yn"):
                break
    return overwrite=="y"