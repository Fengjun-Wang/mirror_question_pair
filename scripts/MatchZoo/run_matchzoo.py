python matchzoo/main.py --phase train --model_file examples/toy_example/config/arci_ranking.config
python matchzoo/main.py --phase predict --model_file examples/toy_example/config/arci_ranking.config

in training, we run:
python matchzoo/main.py --phase train --model_file examples/QuoraQP/config/matchpyramid_quoraqp.config

in testing, we run:
python matchzoo/main.py --phase predict --model_file examples/QuoraQP/config/matchpyramid_quoraqp.config