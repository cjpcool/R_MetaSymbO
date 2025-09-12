# Overall Components
* Generator
![alt text](image-1.png)

## Generator
~~~
python gen_test.py \
  --api_key openai api keys \
  --ckpt-dir ./checkpoints/omat24_rattle2 \
  --prompt "Wide-gap semiconductor, wurtzite-like scaffold" \
  --save-dir ./_gens \
  --device cuda
~~~

## MD Optimize and DFT computation
~~~
python wrap_md_uma_dft.py \
  --gen-path ./_gens/gen_union.npz \
  --ckpt /path/to/uma.pt \
  --run-dft --orca-command orca --nprocs 16 --maxcore 6000 \
  --orcasimpleinput "M062X 6-31G* SP EnGrad D3BJ def2/J RIJCOSX TightSCF NoAutoStart MiniPrint NoPop" \
  --preset thorough \
  --outdir ./_mdopt
~~~

## Analyze MD Optimization Trajectory
~~~
python analyze_optimization.py --root ./_mdopt --pattern "loop_*.traj" --save-dir ./_mdopt/analysis \
  --export-json ./_mdopt/analysis/metrics.json --export-csv ./_mdopt/analysis/metrics.csv --no-show --compare
~~~


~~~
export FAIRCHEM_UMA_CKPT=/path/to/uma.pt
export FAIRCHEM_UMA_CONFIG=/path/to/config.yaml   
export ORCA_COMMAND=/path/to/orca                
python wrap_md_uma_dft.py \
  --gen-path ./_gens/gen_union.npz \
  --run-dft --nprocs 8 --maxcore 4000 \
  --preset standard \
  --outdir ./_mdopt
~~~


