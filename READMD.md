# Deep Learning Lab 5

## Run command

- default

```bash
python dqn-1.py
```

- atari-run

```bash
python dqn-1.py --env-name PongNoFrameskip-v4 --wandb-run-name atari-run --memory-size 200000 --replay-start-size 50000 --batch-size 32 --epsilon-decay 0.999995 --max-episode-steps 10000 --target-update-frequency 1000
```
