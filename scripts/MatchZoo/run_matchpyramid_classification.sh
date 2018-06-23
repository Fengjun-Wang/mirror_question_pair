# cd ../../ we need to change our MatchZoo folder

currpath=`pwd`
# train the model
python matchzoo/main.py --phase train --model_file ${currpath}/examples/toy_example/config/matchpyramid_classify.config


# predict with the model

python matchzoo/main.py --phase predict --model_file ${currpath}/examples/toy_example/config/matchpyramid_classify.config
