if [ "$1" = "cityscapes" ]; then
    if [ "$2" = "" ]; then
        echo 'Invalid username / password'
        exit
    fi

    wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username='$2'&password='$3'&submit=Login' https://www.cityscapes-dataset.com/login/
    wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
    wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
    unzip -qq gtFine_trainvaltest.zip
    unzip -qq leftImg8bit_trainvaltest.zip

elif [ "$1" = "pascal" ]; then
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O VOCtrainval.tar
    tar -xf VOCtrainval.tar

else
    echo "Invalid Argument"
fi
