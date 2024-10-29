# requires: pip install git+https://github.com/rlworkgroup/metaworld.git@18118a28c06893da0f363786696cc792457b062b
MT40_ENVS_v2 = ['reach-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2', 'button-press-topdown-v2',
                'peg-insert-side-v2', 'window-open-v2', 'door-close-v2', 'reach-wall-v2', 'pick-place-wall-v2',
                'button-press-v2', 'button-press-topdown-wall-v2', 'button-press-wall-v2', 'disassemble-v2', 'plate-slide-v2',
                'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'handle-press-v2', 'handle-pull-v2',
                'handle-pull-side-v2', 'stick-push-v2', 'basketball-v2', 'soccer-v2', 'faucet-open-v2', 'coffee-push-v2',
                'coffee-pull-v2', 'coffee-button-v2', 'sweep-v2', 'sweep-into-v2', 'pick-out-of-hole-v2', 'assembly-v2',
                'lever-pull-v2', 'dial-turn-v2', 'bin-picking-v2', 'box-close-v2', 'hand-insert-v2', 'door-lock-v2',
                'door-unlock-v2']

MT50_ENVS_v2 = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2',
                'button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'window-close-v2', 'door-close-v2',
                'reach-wall-v2', 'pick-place-wall-v2', 'push-wall-v2', 'button-press-v2', 'button-press-topdown-wall-v2',
                'button-press-wall-v2', 'peg-unplug-side-v2', 'disassemble-v2', 'hammer-v2', 'plate-slide-v2',
                'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'handle-press-v2',
                'handle-pull-v2', 'handle-press-side-v2', 'handle-pull-side-v2', 'stick-push-v2', 'stick-pull-v2',
                'basketball-v2', 'soccer-v2', 'faucet-open-v2', 'faucet-close-v2', 'coffee-push-v2', 'coffee-pull-v2',
                'coffee-button-v2', 'sweep-v2', 'sweep-into-v2', 'pick-out-of-hole-v2', 'assembly-v2', 'shelf-place-v2',
                'push-back-v2', 'lever-pull-v2', 'dial-turn-v2', 'bin-picking-v2', 'box-close-v2', 'hand-insert-v2',
                'door-lock-v2', 'door-unlock-v2']

MT45_ENVS_v2 = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2',
                'button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'window-close-v2',
                'door-close-v2', 'reach-wall-v2', 'pick-place-wall-v2', 'push-wall-v2', 'button-press-v2',
                'button-press-topdown-wall-v2', 'button-press-wall-v2', 'peg-unplug-side-v2', 'disassemble-v2', 
                'hammer-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 
                'plate-slide-back-side-v2', 'handle-press-v2', 'handle-pull-v2', 'handle-press-side-v2',
                'handle-pull-side-v2', 'stick-push-v2', 'stick-pull-v2', 'basketball-v2', 'soccer-v2',
                'faucet-open-v2', 'faucet-close-v2', 'coffee-push-v2', 'coffee-pull-v2', 'coffee-button-v2',
                'sweep-v2', 'sweep-into-v2', 'pick-out-of-hole-v2', 'assembly-v2', 'shelf-place-v2',
                'push-back-v2', 'lever-pull-v2', 'dial-turn-v2']

MT5_ENVS_v2 = ['bin-picking-v2', 'box-close-v2', 'hand-insert-v2', 'door-lock-v2', 'door-unlock-v2']

CW10_ENVS_V2 = ["hammer-v2", "push-wall-v2", "faucet-close-v2", "push-back-v2", "stick-pull-v2", "handle-press-side-v2",
                "push-v2", "shelf-place-v2", "window-close-v2", "peg-unplug-side-v2"]

PROCGEN_ENVS = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist",
                "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]

PROCGEN_12_ENVS = ["bigfish", "bossfight", "caveflyer", "chaser", "coinrun", "dodgeball", 
                   "fruitbot", "heist", "leaper", "maze", "miner", "starpilot"]

PROCGEN_4_ENVS = ["climber", "ninja", "plunder", "jumper"]


ATARI_NAMES = ['adventure', 'air-raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
              'bank-heist', 'battle-zone', 'beam-rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
              'centipede', 'chopper-command', 'crazy-climber', 'defender', 'demon-attack', 'double-dunk',
              'elevator-action', 'enduro', 'fishing-derby', 'freeway', 'frostbite', 'gopher', 'gravitar', 'hero',
              'ice-hockey', 'jamesbond', 'journey-escape', 'kangaroo', 'krull', 'kung-fu-master',
              'montezuma-revenge', 'ms-pacman', 'name-this-game', 'phoenix', 'pitfall', 'pong', 'pooyan',
              'private-eye', 'qbert', 'riverraid', 'road-runner', 'robotank', 'seaquest', 'skiing', 'solaris',
              'space-invaders', 'star-gunner', 'tennis', 'time-pilot', 'tutankham', 'up-n-down', 'venture',
              'video-pinball', 'wizard-of-wor', 'yars-revenge', 'zaxxon']

ATARI_ENVS = ['AdventureNoFrameskip-v4', 'AirRaidNoFrameskip-v4', 'AlienNoFrameskip-v4', 'AmidarNoFrameskip-v4',
              'AssaultNoFrameskip-v4','AsterixNoFrameskip-v4', 'AsteroidsNoFrameskip-v4', 'AtlantisNoFrameskip-v4',
              'BankHeistNoFrameskip-v4', 'BattleZoneNoFrameskip-v4', 'BeamRiderNoFrameskip-v4', 'BerzerkNoFrameskip-v4',
              'BowlingNoFrameskip-v4', 'BoxingNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'CarnivalNoFrameskip-v4',
              'CentipedeNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4', 
              'DefenderNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'DoubleDunkNoFrameskip-v4', 'ElevatorActionNoFrameskip-v4',
              'EnduroNoFrameskip-v4', 'FishingDerbyNoFrameskip-v4', 'FreewayNoFrameskip-v4', 'FrostbiteNoFrameskip-v4',
              'GopherNoFrameskip-v4', 'GravitarNoFrameskip-v4', 'HeroNoFrameskip-v4', 'IceHockeyNoFrameskip-v4', 
              'JamesbondNoFrameskip-v4', 'JourneyEscapeNoFrameskip-v4', 'KangarooNoFrameskip-v4', 'KrullNoFrameskip-v4', 
              'KungFuMasterNoFrameskip-v4', 'MontezumaRevengeNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 
              'NameThisGameNoFrameskip-v4', 'PhoenixNoFrameskip-v4', 'PitfallNoFrameskip-v4', 'PongNoFrameskip-v4', 
              'PooyanNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'QbertNoFrameskip-v4', 'RiverraidNoFrameskip-v4', 
              'RoadRunnerNoFrameskip-v4', 'RobotankNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'SkiingNoFrameskip-v4',
              'SolarisNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'StarGunnerNoFrameskip-v4', 'TennisNoFrameskip-v4',
              'TimePilotNoFrameskip-v4', 'TutankhamNoFrameskip-v4', 'UpNDownNoFrameskip-v4', 'VentureNoFrameskip-v4',
              'VideoPinballNoFrameskip-v4', 'WizardOfWorNoFrameskip-v4', 'YarsRevengeNoFrameskip-v4', 'ZaxxonNoFrameskip-v4']


ATARI_NAME_TO_ENVID = {name: [env for env in ATARI_ENVS if name.replace("-", "") in env.lower()][0] for name in ATARI_NAMES}

ATARI_ENVID_TO_NAME = {envid: name for name, envid in ATARI_NAME_TO_ENVID.items()}

ATARI_NAMES_46 = ['alien', 'amidar', 'assault', 'asterix', 'atlantis', 'bank-heist', 'battle-zone',
                  'beam-rider', 'boxing', 'breakout', 'carnival', 'centipede', 'chopper-command', 'crazy-climber',
                  'demon-attack', 'double-dunk', 'enduro', 'fishing-derby', 'freeway', 'frostbite', 'gopher',
                  'gravitar', 'hero', 'ice-hockey', 'jamesbond', 'kangaroo', 'krull', 'kung-fu-master', 'ms-pacman', 
                  'name-this-game', 'phoenix', 'pong', 'pooyan', 'qbert', 'riverraid', 'road-runner', 'robotank',
                  'seaquest', 'space-invaders', 'star-gunner','time-pilot','up-n-down', 'video-pinball', 
                  'wizard-of-wor', 'yars-revenge', 'zaxxon']
ATARI_ENVS_46 = [ATARI_NAME_TO_ENVID[name] for name in ATARI_NAMES_46]

# does not contain: alien, pong, ms-pacman, space-invaders, star-gunner
ATARI_NAMES_41 = ['amidar', 'assault', 'asterix', 'atlantis', 'bank-heist', 'battle-zone',
                  'beam-rider', 'boxing', 'breakout', 'carnival', 'centipede', 'chopper-command', 'crazy-climber',
                  'demon-attack', 'double-dunk', 'enduro', 'fishing-derby', 'freeway', 'frostbite', 'gopher',
                  'gravitar', 'hero', 'ice-hockey', 'jamesbond', 'kangaroo', 'krull', 'kung-fu-master', 
                  'name-this-game', 'phoenix', 'pooyan', 'qbert', 'riverraid', 'road-runner', 'robotank',
                  'seaquest','time-pilot','up-n-down', 'video-pinball', 
                  'wizard-of-wor', 'yars-revenge', 'zaxxon']
ATARI_ENVS_41 = [ATARI_NAME_TO_ENVID[name] for name in ATARI_NAMES_41]

ATARI_NAMES_5 = ['alien', 'pong', 'ms-pacman', 'space-invaders', 'star-gunner']
ATARI_ENVS_5 = [ATARI_NAME_TO_ENVID[name] for name in ATARI_NAMES_5]


DM_CONTROL_ENVS = ['acrobot-swingup', 'acrobot-swingup_sparse', 'ball_in_cup-catch', 'cartpole-balance',
                   'cartpole-balance_sparse', 'cartpole-swingup', 'cartpole-swingup_sparse',
                   'cheetah-run', 'finger-spin', 'finger-turn_easy', 'finger-turn_hard', 
                   'fish-upright', 'fish-swim', 'hopper-stand', 'hopper-hop', 'humanoid-stand',
                   'humanoid-walk', 'humanoid-run', 'manipulator-bring_ball', 'pendulum-swingup',
                   'point_mass-easy', 'reacher-easy', 'reacher-hard', 'swimmer-swimmer6', 
                   'swimmer-swimmer15', 'walker-stand', 'walker-walk', 'walker-run',
                   "manipulator-insert_ball", "manipulator-insert_peg"]

DM_CONTROL_ICL_ENVS_11 = ['finger-turn_easy', 'fish-upright', 'hopper-stand',
                          'point_mass-easy', 'walker-stand', 'walker-run', 'ball_in_cup-catch',
                          'cartpole-swingup', 'cheetah-run', 'finger-spin', 'reacher-easy']

DM_CONTROL_ICL_ENVS_5 = ['cartpole-balance', 'finger-turn_hard', "pendulum-swingup",
                         'reacher-hard', 'walker-walk']


DM_CONTROL_ENVID_TO_FILE = {f"{envid}.npz": f"{envid.replace('-', '_')}.npz" for envid in DM_CONTROL_ENVS}

DM_CONTROL_ENVID_TO_DIR = {envid: f"{envid.replace('-', '_')}" for envid in DM_CONTROL_ENVS}

MINIHACK_ENVS = ["MiniHack-Room-Dark-10x10-v0", "MiniHack-Room-Dark-17x17-v0",
                 "MiniHack-Room-Dark-Dense-10x10-v0", "MiniHack-Room-Dark-Dense-17x17-v0",
                 "MiniHack-Room-Dark-Sparse-10x10-v0", "MiniHack-Room-Dark-Sparse-17x17-v0",
                 "MiniHack-Room-Dark-Dense-20x20-v0", "MiniHack-Room-Dark-Dense-40x20-v0", 
                 "MiniHack-KeyDoor-Dark-Dense-10x10-v0", "MiniHack-KeyDoor-Dark-Dense-20x20-v0",
                 "MiniHack-KeyDoor-Dark-Dense-40x20-v0"]

GYM_ENVS = ["Hopper-v3", "HalfCheetah-v3", "Walker2d-v3", "Hopper-v2", "HalfCheetah-v2", "Walker2d-v2"]


COMPOSUITE_ENVS_16 = [
    # include Panda, ObjectWall tasks
    'Panda_Box_ObjectWall_PickPlace', 'Panda_Box_ObjectWall_Push', 'Panda_Box_ObjectWall_Shelf', 'Panda_Box_ObjectWall_Trashcan', 
    'Panda_Dumbbell_ObjectWall_PickPlace', 'Panda_Dumbbell_ObjectWall_Push', 'Panda_Dumbbell_ObjectWall_Shelf', 
    'Panda_Dumbbell_ObjectWall_Trashcan', 'Panda_Plate_ObjectWall_PickPlace', 'Panda_Plate_ObjectWall_Push', 
    'Panda_Plate_ObjectWall_Shelf', 'Panda_Plate_ObjectWall_Trashcan', 'Panda_Hollowbox_ObjectWall_PickPlace', 
    'Panda_Hollowbox_ObjectWall_Push', 'Panda_Hollowbox_ObjectWall_Shelf', 'Panda_Hollowbox_ObjectWall_Trashcan'
]

COMPOSUITE_ENVS_240 = [
    # all except COMPOSUITE_ENVS_16
    'IIWA_Box_None_PickPlace', 'IIWA_Box_None_Push', 'IIWA_Box_None_Shelf', 'IIWA_Box_None_Trashcan', 
    'IIWA_Box_GoalWall_PickPlace', 'IIWA_Box_GoalWall_Push', 'IIWA_Box_GoalWall_Shelf',
    'IIWA_Box_GoalWall_Trashcan', 'IIWA_Box_ObjectDoor_PickPlace', 'IIWA_Box_ObjectDoor_Push',
    'IIWA_Box_ObjectDoor_Shelf', 'IIWA_Box_ObjectDoor_Trashcan', 'IIWA_Box_ObjectWall_PickPlace', 
    'IIWA_Box_ObjectWall_Push', 'IIWA_Box_ObjectWall_Shelf', 'IIWA_Box_ObjectWall_Trashcan', 
    'IIWA_Dumbbell_None_PickPlace', 'IIWA_Dumbbell_None_Push', 'IIWA_Dumbbell_None_Shelf', 
    'IIWA_Dumbbell_None_Trashcan', 'IIWA_Dumbbell_GoalWall_PickPlace', 'IIWA_Dumbbell_GoalWall_Push',
    'IIWA_Dumbbell_GoalWall_Shelf', 'IIWA_Dumbbell_GoalWall_Trashcan', 'IIWA_Dumbbell_ObjectDoor_PickPlace',
    'IIWA_Dumbbell_ObjectDoor_Push', 'IIWA_Dumbbell_ObjectDoor_Shelf', 'IIWA_Dumbbell_ObjectDoor_Trashcan', 
    'IIWA_Dumbbell_ObjectWall_PickPlace', 'IIWA_Dumbbell_ObjectWall_Push', 'IIWA_Dumbbell_ObjectWall_Shelf',
    'IIWA_Dumbbell_ObjectWall_Trashcan', 'IIWA_Plate_None_PickPlace', 'IIWA_Plate_None_Push', 'IIWA_Plate_None_Shelf',
    'IIWA_Plate_None_Trashcan', 'IIWA_Plate_GoalWall_PickPlace', 'IIWA_Plate_GoalWall_Push', 'IIWA_Plate_GoalWall_Shelf',
    'IIWA_Plate_GoalWall_Trashcan', 'IIWA_Plate_ObjectDoor_PickPlace', 'IIWA_Plate_ObjectDoor_Push', 
    'IIWA_Plate_ObjectDoor_Shelf', 'IIWA_Plate_ObjectDoor_Trashcan', 'IIWA_Plate_ObjectWall_PickPlace',
    'IIWA_Plate_ObjectWall_Push', 'IIWA_Plate_ObjectWall_Shelf', 'IIWA_Plate_ObjectWall_Trashcan',
    'IIWA_Hollowbox_None_PickPlace', 'IIWA_Hollowbox_None_Push', 'IIWA_Hollowbox_None_Shelf', 'IIWA_Hollowbox_None_Trashcan',
    'IIWA_Hollowbox_GoalWall_PickPlace', 'IIWA_Hollowbox_GoalWall_Push', 'IIWA_Hollowbox_GoalWall_Shelf',
    'IIWA_Hollowbox_GoalWall_Trashcan', 'IIWA_Hollowbox_ObjectDoor_PickPlace', 'IIWA_Hollowbox_ObjectDoor_Push',
    'IIWA_Hollowbox_ObjectDoor_Shelf', 'IIWA_Hollowbox_ObjectDoor_Trashcan', 'IIWA_Hollowbox_ObjectWall_PickPlace', 
    'IIWA_Hollowbox_ObjectWall_Push', 'IIWA_Hollowbox_ObjectWall_Shelf', 'IIWA_Hollowbox_ObjectWall_Trashcan', 
    'Jaco_Box_None_PickPlace', 'Jaco_Box_None_Push', 'Jaco_Box_None_Shelf', 'Jaco_Box_None_Trashcan', 'Jaco_Box_GoalWall_PickPlace',
    'Jaco_Box_GoalWall_Push', 'Jaco_Box_GoalWall_Shelf', 'Jaco_Box_GoalWall_Trashcan', 'Jaco_Box_ObjectDoor_PickPlace', 
    'Jaco_Box_ObjectDoor_Push', 'Jaco_Box_ObjectDoor_Shelf', 'Jaco_Box_ObjectDoor_Trashcan', 'Jaco_Box_ObjectWall_PickPlace', 
    'Jaco_Box_ObjectWall_Push', 'Jaco_Box_ObjectWall_Shelf', 'Jaco_Box_ObjectWall_Trashcan', 'Jaco_Dumbbell_None_PickPlace',
    'Jaco_Dumbbell_None_Push', 'Jaco_Dumbbell_None_Shelf', 'Jaco_Dumbbell_None_Trashcan', 'Jaco_Dumbbell_GoalWall_PickPlace',
    'Jaco_Dumbbell_GoalWall_Push', 'Jaco_Dumbbell_GoalWall_Shelf', 'Jaco_Dumbbell_GoalWall_Trashcan', 
    'Jaco_Dumbbell_ObjectDoor_PickPlace', 'Jaco_Dumbbell_ObjectDoor_Push', 'Jaco_Dumbbell_ObjectDoor_Shelf', 
    'Jaco_Dumbbell_ObjectDoor_Trashcan', 'Jaco_Dumbbell_ObjectWall_PickPlace', 'Jaco_Dumbbell_ObjectWall_Push',
    'Jaco_Dumbbell_ObjectWall_Shelf', 'Jaco_Dumbbell_ObjectWall_Trashcan', 'Jaco_Plate_None_PickPlace', 'Jaco_Plate_None_Push',
    'Jaco_Plate_None_Shelf', 'Jaco_Plate_None_Trashcan', 'Jaco_Plate_GoalWall_PickPlace', 'Jaco_Plate_GoalWall_Push',
    'Jaco_Plate_GoalWall_Shelf', 'Jaco_Plate_GoalWall_Trashcan', 'Jaco_Plate_ObjectDoor_PickPlace', 'Jaco_Plate_ObjectDoor_Push', 
    'Jaco_Plate_ObjectDoor_Shelf', 'Jaco_Plate_ObjectDoor_Trashcan', 'Jaco_Plate_ObjectWall_PickPlace', 
    'Jaco_Plate_ObjectWall_Push', 'Jaco_Plate_ObjectWall_Shelf', 'Jaco_Plate_ObjectWall_Trashcan', 
    'Jaco_Hollowbox_None_PickPlace', 'Jaco_Hollowbox_None_Push', 'Jaco_Hollowbox_None_Shelf', 'Jaco_Hollowbox_None_Trashcan',
    'Jaco_Hollowbox_GoalWall_PickPlace', 'Jaco_Hollowbox_GoalWall_Push', 'Jaco_Hollowbox_GoalWall_Shelf', 
    'Jaco_Hollowbox_GoalWall_Trashcan', 'Jaco_Hollowbox_ObjectDoor_PickPlace', 'Jaco_Hollowbox_ObjectDoor_Push',
    'Jaco_Hollowbox_ObjectDoor_Shelf', 'Jaco_Hollowbox_ObjectDoor_Trashcan', 'Jaco_Hollowbox_ObjectWall_PickPlace',
    'Jaco_Hollowbox_ObjectWall_Push', 'Jaco_Hollowbox_ObjectWall_Shelf', 'Jaco_Hollowbox_ObjectWall_Trashcan', 
    'Kinova3_Box_None_PickPlace', 'Kinova3_Box_None_Push', 'Kinova3_Box_None_Shelf', 'Kinova3_Box_None_Trashcan',
    'Kinova3_Box_GoalWall_PickPlace', 'Kinova3_Box_GoalWall_Push', 'Kinova3_Box_GoalWall_Shelf', 
    'Kinova3_Box_GoalWall_Trashcan', 'Kinova3_Box_ObjectDoor_PickPlace', 'Kinova3_Box_ObjectDoor_Push', 
    'Kinova3_Box_ObjectDoor_Shelf', 'Kinova3_Box_ObjectDoor_Trashcan', 'Kinova3_Box_ObjectWall_PickPlace', 
    'Kinova3_Box_ObjectWall_Push', 'Kinova3_Box_ObjectWall_Shelf', 'Kinova3_Box_ObjectWall_Trashcan', 
    'Kinova3_Dumbbell_None_PickPlace', 'Kinova3_Dumbbell_None_Push', 'Kinova3_Dumbbell_None_Shelf', 
    'Kinova3_Dumbbell_None_Trashcan', 'Kinova3_Dumbbell_GoalWall_PickPlace', 'Kinova3_Dumbbell_GoalWall_Push',
    'Kinova3_Dumbbell_GoalWall_Shelf', 'Kinova3_Dumbbell_GoalWall_Trashcan', 'Kinova3_Dumbbell_ObjectDoor_PickPlace',
    'Kinova3_Dumbbell_ObjectDoor_Push', 'Kinova3_Dumbbell_ObjectDoor_Shelf', 'Kinova3_Dumbbell_ObjectDoor_Trashcan', 
    'Kinova3_Dumbbell_ObjectWall_PickPlace', 'Kinova3_Dumbbell_ObjectWall_Push', 'Kinova3_Dumbbell_ObjectWall_Shelf',
    'Kinova3_Dumbbell_ObjectWall_Trashcan', 'Kinova3_Plate_None_PickPlace', 'Kinova3_Plate_None_Push', 'Kinova3_Plate_None_Shelf', 
    'Kinova3_Plate_None_Trashcan', 'Kinova3_Plate_GoalWall_PickPlace', 'Kinova3_Plate_GoalWall_Push', 
    'Kinova3_Plate_GoalWall_Shelf', 'Kinova3_Plate_GoalWall_Trashcan', 'Kinova3_Plate_ObjectDoor_PickPlace',
    'Kinova3_Plate_ObjectDoor_Push', 'Kinova3_Plate_ObjectDoor_Shelf', 'Kinova3_Plate_ObjectDoor_Trashcan', 
    'Kinova3_Plate_ObjectWall_PickPlace', 'Kinova3_Plate_ObjectWall_Push', 'Kinova3_Plate_ObjectWall_Shelf', 
    'Kinova3_Plate_ObjectWall_Trashcan', 'Kinova3_Hollowbox_None_PickPlace', 'Kinova3_Hollowbox_None_Push', 
    'Kinova3_Hollowbox_None_Shelf', 'Kinova3_Hollowbox_None_Trashcan', 'Kinova3_Hollowbox_GoalWall_PickPlace',
    'Kinova3_Hollowbox_GoalWall_Push', 'Kinova3_Hollowbox_GoalWall_Shelf', 'Kinova3_Hollowbox_GoalWall_Trashcan',
    'Kinova3_Hollowbox_ObjectDoor_PickPlace', 'Kinova3_Hollowbox_ObjectDoor_Push', 'Kinova3_Hollowbox_ObjectDoor_Shelf',
    'Kinova3_Hollowbox_ObjectDoor_Trashcan', 'Kinova3_Hollowbox_ObjectWall_PickPlace', 'Kinova3_Hollowbox_ObjectWall_Push',
    'Kinova3_Hollowbox_ObjectWall_Shelf', 'Kinova3_Hollowbox_ObjectWall_Trashcan', 'Panda_Box_None_PickPlace', 
    'Panda_Box_None_Push', 'Panda_Box_None_Shelf', 'Panda_Box_None_Trashcan', 'Panda_Box_GoalWall_PickPlace',
    'Panda_Box_GoalWall_Push', 'Panda_Box_GoalWall_Shelf', 'Panda_Box_GoalWall_Trashcan', 'Panda_Box_ObjectDoor_PickPlace',
    'Panda_Box_ObjectDoor_Push', 'Panda_Box_ObjectDoor_Shelf', 'Panda_Box_ObjectDoor_Trashcan', 
    'Panda_Dumbbell_None_PickPlace', 'Panda_Dumbbell_None_Push', 'Panda_Dumbbell_None_Shelf', 'Panda_Dumbbell_None_Trashcan',
    'Panda_Dumbbell_GoalWall_PickPlace', 'Panda_Dumbbell_GoalWall_Push', 'Panda_Dumbbell_GoalWall_Shelf',
    'Panda_Dumbbell_GoalWall_Trashcan', 'Panda_Dumbbell_ObjectDoor_PickPlace', 'Panda_Dumbbell_ObjectDoor_Push',
    'Panda_Dumbbell_ObjectDoor_Shelf', 'Panda_Dumbbell_ObjectDoor_Trashcan', 'Panda_Plate_None_PickPlace', 
    'Panda_Plate_None_Push', 'Panda_Plate_None_Shelf', 'Panda_Plate_None_Trashcan', 'Panda_Plate_GoalWall_PickPlace',
    'Panda_Plate_GoalWall_Push', 'Panda_Plate_GoalWall_Shelf', 'Panda_Plate_GoalWall_Trashcan',
    'Panda_Plate_ObjectDoor_PickPlace', 'Panda_Plate_ObjectDoor_Push', 'Panda_Plate_ObjectDoor_Shelf',
    'Panda_Plate_ObjectDoor_Trashcan', 'Panda_Hollowbox_None_PickPlace', 'Panda_Hollowbox_None_Push', 
    'Panda_Hollowbox_None_Shelf', 'Panda_Hollowbox_None_Trashcan', 'Panda_Hollowbox_GoalWall_PickPlace',
    'Panda_Hollowbox_GoalWall_Push', 'Panda_Hollowbox_GoalWall_Shelf', 'Panda_Hollowbox_GoalWall_Trashcan', 
    'Panda_Hollowbox_ObjectDoor_PickPlace', 'Panda_Hollowbox_ObjectDoor_Push', 'Panda_Hollowbox_ObjectDoor_Shelf', 
    'Panda_Hollowbox_ObjectDoor_Trashcan'
]

COMPOSUITE_ENVS = [*COMPOSUITE_ENVS_16, *COMPOSUITE_ENVS_240]

MIMICGEN_ENVS_24 = [
    'CoffeePreparation_D0', 'CoffeePreparation_D1', 'Coffee_D0', 'Coffee_D1', 'Coffee_D2', 'HammerCleanup_D0', 
    'HammerCleanup_D1', 'Kitchen_D0', 'Kitchen_D1', 'MugCleanup_D0', 'MugCleanup_D1', 'NutAssembly_D0', 'PickPlace_D0',
    'Square_D0', 'Square_D1', 'Square_D2', 'StackThree_D0', "StackThree_D1", 'Stack_D0', "Stack_D1",
    'Threading_D0', 'Threading_D1', 'ThreePieceAssembly_D0', 'ThreePieceAssembly_D1',     
]

MIMICGEN_ENVS_83 = [
    # original mimicgen
    'CoffeePreparation_D0', 'CoffeePreparation_D1', 'Coffee_D0', 'Coffee_D1', 'Coffee_D2', 'HammerCleanup_D0', 
    'HammerCleanup_D1', 'Kitchen_D0', 'Kitchen_D1', 'MugCleanup_D0', 'MugCleanup_D1', 'NutAssembly_D0', 'PickPlace_D0',
    'Square_D0', 'Square_D1', 'Square_D2', 'StackThree_D0', "StackThree_D1", 'Stack_D0', "Stack_D1",
    'Threading_D0', 'Threading_D1', 'ThreePieceAssembly_D0', 'ThreePieceAssembly_D1',      
    # self-generated mimicgen with additional arms
    'Coffee_D0-IIWA',  'Coffee_D0-Sawyer',  'Coffee_D0-UR5e', 
    'Coffee_D1-IIWA', 'Coffee_D1-Sawyer', 'Coffee_D1-UR5e',
    'Coffee_D2-IIWA', 'Coffee_D2-UR5e',
    'HammerCleanup_D0-IIWA',  'HammerCleanup_D0-Sawyer',  'HammerCleanup_D0-UR5e', 
    'HammerCleanup_D1-IIWA', 'HammerCleanup_D1-Sawyer', 'HammerCleanup_D1-UR5e',
    'Kitchen_D0-IIWA', 'Kitchen_D0-UR5e',
    'Kitchen_D1-UR5e', 
    'MugCleanup_D0-IIWA',
    'MugCleanup_D1-IIWA', 'MugCleanup_D1-UR5e',
    'NutAssembly_D0-IIWA',  'NutAssembly_D0-Sawyer',  'NutAssembly_D0-UR5e', 
    'PickPlace_D0-IIWA', 'PickPlace_D0-Sawyer', 'PickPlace_D0-UR5e',
    'Square_D0-IIWA', 'Square_D0-Sawyer', 'Square_D0-UR5e',
    'Square_D1-IIWA',  'Square_D1-Sawyer',  'Square_D1-UR5e', 
    'StackThree_D0-IIWA', 'StackThree_D0-Sawyer', 'StackThree_D0-UR5e',
    "StackThree_D1-IIWA",  "StackThree_D1-Sawyer",  "StackThree_D1-UR5e", 
    'Stack_D0-IIWA', 'Stack_D0-Sawyer', 'Stack_D0-UR5e',
    "Stack_D1-IIWA", "Stack_D1-Sawyer", "Stack_D1-UR5e",
    'Threading_D0-IIWA',  'Threading_D0-Sawyer',  'Threading_D0-UR5e', 
    'Threading_D1-IIWA',  'Threading_D1-Sawyer',  'Threading_D1-UR5e', 
    'ThreePieceAssembly_D0-IIWA', 'ThreePieceAssembly_D0-Sawyer', 'ThreePieceAssembly_D0-UR5e',
    'ThreePieceAssembly_D1-IIWA', 'ThreePieceAssembly_D1-Sawyer', 'ThreePieceAssembly_D1-UR5e',   
    'ThreePieceAssembly_D2-IIWA', 'ThreePieceAssembly_D2-Sawyer', 'ThreePieceAssembly_D2-UR5e',   
]

MIMICGEN_ENVS_2 = ['Threading_D2', 'ThreePieceAssembly_D2']

MIMICGEN_ENVS = [*MIMICGEN_ENVS_24, *MIMICGEN_ENVS_2, *MIMICGEN_ENVS_83]
for key in [*MIMICGEN_ENVS_24, *MIMICGEN_ENVS_2]:
    for robot in ["Panda", "Sawyer", "IIWA", "UR5e"]:
        MIMICGEN_ENVS.append(f"{robot}_{key}")

MIMICGEN_NAME_TO_ENVID = {
    "coffee_d0": "Coffee_D0",
    "coffee_d1": "Coffee_D1",
    "coffee_d2": "Coffee_D2",
    "coffee_preparation_d0": "CoffeePreparation_D0",
    "coffee_preparation_d1": "CoffeePreparation_D1",
    "hammer_cleanup_d0": "HammerCleanup_D0",
    "hammer_cleanup_d1": "HammerCleanup_D1",
    "kitchen_d0": "Kitchen_D0",
    "kitchen_d1": "Kitchen_D1",
    "mug_cleanup_d0": "MugCleanup_D0",
    "mug_cleanup_d1": "MugCleanup_D1",
    "nut_assembly_d0": "NutAssembly_D0",
    "pick_place_d0": "PickPlace_D0",
    "square_d0": "Square_D0",
    "square_d1": "Square_D1",
    "square_d2": "Square_D2",
    "stack_d0": "Stack_D0",
    "stack_d1": "Stack_D1",
    "stack_three_d0": "StackThree_D0",
    "stack_three_d1": "StackThree_D1",
    "threading_d0": "Threading_D0",
    "threading_d1": "Threading_D1",
    "three_piece_assembly_d0": "ThreePieceAssembly_D0",
    "three_piece_assembly_d1": "ThreePieceAssembly_D1",
    # self-generate mimicgen data for other robots
    "coffee_D0_robot_IIWA_gripper_Robotiq85Gripper": "Coffee_D0-IIWA",
    "coffee_D0_robot_Sawyer_gripper_RethinkGripper": "Coffee_D0-Sawyer",
    "coffee_D0_robot_UR5e_gripper_Robotiq85Gripper": "Coffee_D0-UR5e",
    "coffee_D1_robot_IIWA_gripper_Robotiq85Gripper": "Coffee_D1-IIWA",
    "coffee_D1_robot_Sawyer_gripper_RethinkGripper": "Coffee_D1-Sawyer",
    "coffee_D1_robot_UR5e_gripper_Robotiq85Gripper": "Coffee_D1-UR5e",
    "coffee_D2_robot_IIWA_gripper_Robotiq85Gripper": "Coffee_D2-IIWA",
    "coffee_D2_robot_UR5e_gripper_Robotiq85Gripper": "Coffee_D2-UR5e",
    "hammer_cleanup_D0_robot_IIWA_gripper_Robotiq85Gripper": "HammerCleanup_D0-IIWA",
    "hammer_cleanup_D0_robot_Sawyer_gripper_RethinkGripper": "HammerCleanup_D0-Sawyer",
    "hammer_cleanup_D0_robot_UR5e_gripper_Robotiq85Gripper": "HammerCleanup_D0-UR5e",
    "hammer_cleanup_D1_robot_IIWA_gripper_Robotiq85Gripper": "HammerCleanup_D1-IIWA",
    "hammer_cleanup_D1_robot_Sawyer_gripper_RethinkGripper": "HammerCleanup_D1-Sawyer",
    "hammer_cleanup_D1_robot_UR5e_gripper_Robotiq85Gripper": "HammerCleanup_D1-UR5e",
    "kitchen_D0_robot_IIWA_gripper_Robotiq85Gripper": "Kitchen_D0-IIWA",
    "kitchen_D0_robot_UR5e_gripper_Robotiq85Gripper": "Kitchen_D0-UR5e",
    "kitchen_D1_robot_UR5e_gripper_Robotiq85Gripper": "Kitchen_D1-UR5e",
    "mug_cleanup_D0_robot_IIWA_gripper_Robotiq85Gripper": "MugCleanup_D0-IIWA",
    "mug_cleanup_D1_robot_IIWA_gripper_Robotiq85Gripper": "MugCleanup_D1-IIWA",
    "mug_cleanup_D1_robot_UR5e_gripper_Robotiq85Gripper": "MugCleanup_D1-UR5e",
    "nut_assembly_D0_robot_IIWA_gripper_Robotiq85Gripper": "NutAssembly_D0-IIWA",
    "nut_assembly_D0_robot_Sawyer_gripper_RethinkGripper": "NutAssembly_D0-Sawyer",
    "nut_assembly_D0_robot_UR5e_gripper_Robotiq85Gripper": "NutAssembly_D0-UR5e",
    "pick_place_D0_robot_IIWA_gripper_Robotiq85Gripper": "PickPlace_D0-IIWA",
    "pick_place_D0_robot_Sawyer_gripper_RethinkGripper": "PickPlace_D0-Sawyer",
    "pick_place_D0_robot_UR5e_gripper_Robotiq85Gripper": "PickPlace_D0-UR5e",
    "square_D0_robot_IIWA_gripper_Robotiq85Gripper": "Square_D0-IIWA",
    "square_D0_robot_Sawyer_gripper_RethinkGripper": "Square_D0-Sawyer",
    "square_D0_robot_UR5e_gripper_Robotiq85Gripper": "Square_D0-UR5e",
    "square_D1_robot_IIWA_gripper_Robotiq85Gripper": "Square_D1-IIWA",
    "square_D1_robot_Sawyer_gripper_RethinkGripper": "Square_D1-Sawyer",
    "square_D1_robot_UR5e_gripper_Robotiq85Gripper": "Square_D1-UR5e",
    "stack_D0_robot_IIWA_gripper_Robotiq85Gripper": "Stack_D0-IIWA",
    "stack_D0_robot_Sawyer_gripper_RethinkGripper": "Stack_D0-Sawyer",
    "stack_D0_robot_UR5e_gripper_Robotiq85Gripper": "Stack_D0-UR5e",
    "stack_D1_robot_IIWA_gripper_Robotiq85Gripper": "Stack_D1-IIWA",
    "stack_D1_robot_Sawyer_gripper_RethinkGripper": "Stack_D1-Sawyer",
    "stack_D1_robot_UR5e_gripper_Robotiq85Gripper": "Stack_D1-UR5e",
    "stack_three_D0_robot_IIWA_gripper_Robotiq85Gripper": "StackThree_D0-IIWA",
    "stack_three_D0_robot_Sawyer_gripper_RethinkGripper": "StackThree_D0-Sawyer",
    "stack_three_D0_robot_UR5e_gripper_Robotiq85Gripper": "StackThree_D0-UR5e",
    "stack_three_D1_robot_IIWA_gripper_Robotiq85Gripper": "StackThree_D1-IIWA",
    "stack_three_D1_robot_Sawyer_gripper_RethinkGripper": "StackThree_D1-Sawyer",
    "stack_three_D1_robot_UR5e_gripper_Robotiq85Gripper": "StackThree_D1-UR5e",
    "threading_D0_robot_IIWA_gripper_Robotiq85Gripper": "Threading_D0-IIWA",
    "threading_D0_robot_Sawyer_gripper_RethinkGripper": "Threading_D0-Sawyer",
    "threading_D0_robot_UR5e_gripper_Robotiq85Gripper": "Threading_D0-UR5e",
    "threading_D1_robot_IIWA_gripper_Robotiq85Gripper": "Threading_D1-IIWA",
    "threading_D1_robot_Sawyer_gripper_RethinkGripper": "Threading_D1-Sawyer",
    "threading_D1_robot_UR5e_gripper_Robotiq85Gripper": "Threading_D1-UR5e",
    "three_piece_assembly_D0_robot_IIWA_gripper_Robotiq85Gripper": "ThreePieceAssembly_D0-IIWA",
    "three_piece_assembly_D0_robot_Sawyer_gripper_RethinkGripper": "ThreePieceAssembly_D0-Sawyer",
    "three_piece_assembly_D0_robot_UR5e_gripper_Robotiq85Gripper": "ThreePieceAssembly_D0-UR5e",
    "three_piece_assembly_D1_robot_IIWA_gripper_Robotiq85Gripper": "ThreePieceAssembly_D1-IIWA",
    "three_piece_assembly_D1_robot_Sawyer_gripper_RethinkGripper": "ThreePieceAssembly_D1-Sawyer",
    "three_piece_assembly_D1_robot_UR5e_gripper_Robotiq85Gripper": "ThreePieceAssembly_D1-UR5e",
    "three_piece_assembly_D2_robot_IIWA_gripper_Robotiq85Gripper": "ThreePieceAssembly_D2-IIWA",
    "three_piece_assembly_D2_robot_Sawyer_gripper_RethinkGripper": "ThreePieceAssembly_D2-Sawyer",
    "three_piece_assembly_D2_robot_UR5e_gripper_Robotiq85Gripper": "ThreePieceAssembly_D2-UR5e",
}

MIMICGEN_ENVID_TO_NAME = {v: k for k, v in MIMICGEN_NAME_TO_ENVID.items()}


ID_TO_NAMES = {
    "mt50_v2": MT50_ENVS_v2,
    "mt40_v2": MT40_ENVS_v2,
    "mt45_v2": MT45_ENVS_v2,
    "mt5_v2": MT5_ENVS_v2,
    "cw10_v2": CW10_ENVS_V2,
    "atari46": ATARI_ENVS_46,
    "atari41": ATARI_ENVS_41, 
    "atari46_mt40v2": [*ATARI_ENVS_46, *MT40_ENVS_v2],
    "atari41_mt40v2": [*ATARI_ENVS_41, *MT40_ENVS_v2], 
    "dmcontrol": DM_CONTROL_ENVS,
    "dmcontrol11_icl": DM_CONTROL_ICL_ENVS_11,
    "dmcontrol5_icl": DM_CONTROL_ICL_ENVS_5,
    "atari5": ATARI_ENVS_5,
    "cw10v2_atari5": [*CW10_ENVS_V2, *ATARI_ENVS_5], 
    "procgen16": PROCGEN_ENVS,
    "procgen12": PROCGEN_12_ENVS,
    "procgen4": PROCGEN_4_ENVS,
    "mt45v2_dmc11": [*MT45_ENVS_v2, *DM_CONTROL_ICL_ENVS_11], 
    "mt5v2_dmc5": [*MT5_ENVS_v2, *DM_CONTROL_ICL_ENVS_5], 
    "mt45v2_dmc11_pg12": [*MT45_ENVS_v2, *DM_CONTROL_ICL_ENVS_11, *PROCGEN_12_ENVS], 
    "mt5v2_dmc5_pg4": [*MT5_ENVS_v2, *DM_CONTROL_ICL_ENVS_5, *PROCGEN_4_ENVS], 
    "mt45v2_dmc11_pg12_atari41": [*MT45_ENVS_v2, *DM_CONTROL_ICL_ENVS_11, *PROCGEN_12_ENVS, *ATARI_ENVS_41], 
    "mt5v2_dmc5_pg4_atari5": [*MT5_ENVS_v2, *DM_CONTROL_ICL_ENVS_5, *PROCGEN_4_ENVS, *ATARI_ENVS_5], 
    "composuite": COMPOSUITE_ENVS,
    "composuite16": COMPOSUITE_ENVS_16,
    "composuite240": COMPOSUITE_ENVS_240,
    "mt45v2_dmc11_pg12_atari41_cs240": [*MT45_ENVS_v2, *DM_CONTROL_ICL_ENVS_11, *PROCGEN_12_ENVS,
                                        *ATARI_ENVS_41, *COMPOSUITE_ENVS_240], 
    "mt5v2_dmc5_pg4_atari5_cs16": [*MT5_ENVS_v2, *DM_CONTROL_ICL_ENVS_5, *PROCGEN_4_ENVS,
                                   *ATARI_ENVS_5, *COMPOSUITE_ENVS_16], 
    "mimicgen": MIMICGEN_ENVS,
    "mimicgen24": MIMICGEN_ENVS_24,
    "mimicgen83": MIMICGEN_ENVS_83,
    "mimicgen2": MIMICGEN_ENVS_2,
    "mt45v2_dmc11_pg12_atari41_cs240_mg24": [*MT45_ENVS_v2, *DM_CONTROL_ICL_ENVS_11, *PROCGEN_12_ENVS,
                                             *ATARI_ENVS_41, *COMPOSUITE_ENVS_240, *MIMICGEN_ENVS_24], 
    "mt5v2_dmc5_pg4_atari5_cs16_mg2": [*MT5_ENVS_v2, *DM_CONTROL_ICL_ENVS_5, *PROCGEN_4_ENVS,
                                       *ATARI_ENVS_5, *COMPOSUITE_ENVS_16, *MIMICGEN_ENVS_2], 
    "mt45v2_dmc11_pg12_atari41_cs240_mg83": [*MT45_ENVS_v2, *DM_CONTROL_ICL_ENVS_11, *PROCGEN_12_ENVS,
                                             *ATARI_ENVS_41, *COMPOSUITE_ENVS_240, *MIMICGEN_ENVS_83],
}


ID_TO_DOMAIN = {
    **{envid: "dmcontrol" for envid in [*DM_CONTROL_ENVS, "dmcontrol11_icl", "dmcontrol5_icl", "dmcontrol6_icl", "dmcontrol4_icl"]},
    **{envid: "atari" for envid in [*ATARI_ENVS_46, *ATARI_NAMES]},
    **{envid: "procgen" for envid in [*PROCGEN_ENVS, "procgen16", "procgen12", "procgen4"]},
    **{envid: "mt50" for envid in [*MT50_ENVS_v2,
                                   "mt40", "mt50", "mt40_v2", "mt50_v2", 
                                   "mt45_v2", "mt5_v2", "mt45_easy_v2", "mt5_easy_v2",
                                   "ml10_v2", "ml5_v2", "mt5_small_v2", "mt5_small_eval_v2"]},
    **{envid: "cw10" for envid in ["cw10", "cw20", "cw10_v2", "cw20_v2"]},
    **{envid: "composuite" for envid in [*COMPOSUITE_ENVS, "composuite16", "composuite240"]},
    **{envid: "mimicgen" for envid in [*MIMICGEN_ENVS, "mimicgen24", "mimicgen2", "mimicgen83"]},
    **{"Dummy-v1": "dummy"}
}

ENVID_TO_NAME = {**ATARI_ENVID_TO_NAME, **DM_CONTROL_ENVID_TO_FILE, **DM_CONTROL_ENVID_TO_DIR, **MIMICGEN_ENVID_TO_NAME}

NAME_TO_ENVID = {v:k for k, v in ENVID_TO_NAME.items()}
