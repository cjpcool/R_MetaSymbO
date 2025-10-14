# Tutorial for access to fastapi servers
After I started the server in the open port, it is easy to access and run the model via curl.

## Pipeline Running
~~~
curl -X POST http://zhoulab-1.cs.vt.edu:5560/pipeline -H "Content-Type: application/json" -d '{
    "claims": "Al20Zn80 at 870K is a solid at equilibrium",
    "ckpt_dir": "./checkpoints/omat24_rattle2",
    "use_llm": true,
    "save_dir": "./_gens",
    "no_generator": true,
    "api_key": "openai api key",
    "no-generator": true,

    "gen_path": "./_gens/gen_union.npz",
    "outdir": "./_mdopt",
    "run_dft": true,
    "uma_ckpt": "/home/grads/jianpengc/projects/materials/R_MetaSymbO/checkpoints/uma-s-1p1.pt",
    "loops": 2,
    "preset": "sprint" / "quick" / "standard",
    "orca_command": "/home/grads/jianpengc/orca_6_1_0/orca",
    "orca_maxcore": 4000,
    "orca_nprocs": 8
  }'
~~~


## Step by Step Running
Command for Step 1:

~~~
curl -X POST http://localhost:5560/generate \
  -H "Content-Type: application/json" \
  -d '{
    "claims": "Al20Zn80 at 870K is a solid at equilibrium",
    "ckpt_dir": "./checkpoints/omat24_rattle2",
    "use_llm": true,
    "save_dir": "./_gens",
    "no_generator": true,
    "api_key": "openai api key",
    "uma"
  }'
  ~~~

Command for Step 2 and 3 (uma optimize + dft computation):
~~~
curl -X POST http://localhost:5560/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "gen_path": "./_gens/gen_union.npz",
    "outdir": "./_mdopt",
    "run_dft": true,
    "ckpt": "/home/grads/jianpengc/projects/materials/R_MetaSymbO/checkpoints/uma-s-1p1.pt",
    "loops": 2,
    "preset": ["quick","sprint", "standard", "thorough"],
    "orca_command": "/home/grads/jianpengc/orca_6_1_0/orca",
    "orca_maxcore": 4000,
    "orca_nprocs": 8
  }'
~~~

