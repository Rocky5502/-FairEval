import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ============================================================
# FairEval White-box Campaign Runner V2
# Resume-aware
# ECIR2027
# ============================================================


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_gpu():

    info = {
        "cuda": torch.cuda.is_available()
    }

    if torch.cuda.is_available():

        p = torch.cuda.get_device_properties(0)

        info.update(
            {
                "name": torch.cuda.get_device_name(0),
                "vram_gb": round(
                    p.total_memory / 1024**3,
                    2
                )
            }
        )

    return info



def load_model(model_id):

    print("="*80)
    print("Loading:", model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )


    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quant,
        trust_remote_code=True
    )


    model.eval()

    return tokenizer, model



def load_existing(path):

    completed=set()

    if not path.exists():
        return completed


    with open(path,"r",encoding="utf-8") as f:

        for line in f:

            try:

                x=json.loads(line)

                completed.add(
                    (
                        x["seed"],
                        x["user"]
                    )
                )

            except:
                pass


    return completed



def generate(
        tokenizer,
        model,
        prompt
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():

        output=model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )


    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )



def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--config",
        required=True
    )

    parser.add_argument(
        "--users",
        type=int,
        default=360
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1]
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )


    args=parser.parse_args()



    cfg=load_yaml(args.config)


    outdir=Path(
        "results/whitebox"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )


    manifest={

        "time":time.time(),

        "gpu":get_gpu(),

        "users":args.users,

        "seeds":args.seeds

    }


    with open(
        outdir/"campaign_manifest.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )



    for name,info in cfg["models"].items():


        outfile=outdir/f"{name}_outputs.jsonl"


        done_file=outdir/f"{name}.done.json"



        existing=load_existing(outfile)



        expected=args.users*len(args.seeds)



        print("\n")
        print("="*80)
        print(
            "MODEL:",
            name
        )


        print(
            "Existing:",
            len(existing),
            "/",
            expected
        )


        if args.resume and len(existing)>=expected:

            print(
                "SKIP:",
                name,
                "already complete"
            )


            continue



        tokenizer,model=load_model(
            info["hf_id"]
        )



        with open(
            outfile,
            "a",
            encoding="utf-8"
        ) as f:



            for seed in args.seeds:


                torch.manual_seed(seed)



                for user in range(args.users):


                    key=(seed,user)


                    if key in existing:

                        continue



                    prompt=f"""
You are evaluating a recommendation assistant.

User ID:
{user}

Provide a helpful recommendation.
"""


                    answer=generate(
                        tokenizer,
                        model,
                        prompt
                    )


                    record={

                        "model":name,

                        "seed":seed,

                        "user":user,

                        "response":answer

                    }


                    f.write(
                        json.dumps(record)
                        +"\n"
                    )

                    f.flush()



                    if user%10==0:

                        print(
                            name,
                            "seed",
                            seed,
                            "user",
                            user
                        )



        with open(
            done_file,
            "w"
        ) as f:

            json.dump(
                {
                    "completed":True,
                    "time":time.time()
                },
                f,
                indent=2
            )


        del model

        torch.cuda.empty_cache()



if __name__=="__main__":
    main()