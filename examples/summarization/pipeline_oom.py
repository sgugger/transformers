import time

import torch
import argparse
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline

HF_MODEL = "HuggingFace"
PIPELINE_MODEL = "Pipeline"

class HuggingFace():
    def __init__(self, fp16=True):
        self.bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", output_past=True)
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.bart.cuda()
        self.bart.eval()
        if fp16:
            self.bart.half()

    def predict(self, samples):
        inputs = self.tokenizer.batch_encode_plus(samples,
                                                  max_length=self.bart.config.max_position_embeddings,
                                                  pad_to_max_length=True,
                                                  return_tensors="pt")

        with torch.no_grad():
            hypo = self.bart.generate(input_ids=inputs["input_ids"].cuda(),
                                      attention_mask=inputs["attention_mask"].cuda(),
                                      num_beams=4,
                                      length_penalty=2.0,
                                      max_length=140,
                                      min_len=55,
                                      no_repeat_ngram_size=3)
        return [self.tokenizer.decode(h, skip_special_tokens=True) for h in hypo]

class Pipeline():
    def __init__(self, fp16=True):
        self.pipeline = pipeline(task='summarization', model='facebook/bart-large-cnn', device=0)
        if fp16:
            self.pipeline.model.half()

    def predict(self, samples):
        with torch.no_grad():
            return self.pipeline(samples, num_beams=4, length_penalty=2.0,
                                 max_length=140, min_len=55, no_repeat_ngram_size=3)

def _get_samples(file):
    with open(file) as source:
        for line in source:
            yield line.strip()

def timed_predict(model, batch):
    t0 = time.time()
    model.predict(batch)
    t1 = time.time()
    return t1 - t0

def main(args):
    if args.model == HF_MODEL:
        model = HuggingFace(fp16=not args.fp32)
    elif args.model == PIPELINE_MODEL:
        model = Pipeline(fp16=not args.fp32)

    count_sample = 0
    count_batch = 0
    batch = []
    tot_time = 0
    for sample in tqdm(_get_samples(args.source), total=args.samples):
        count_sample += 1
        batch.append(sample)

        if len(batch) % args.batch_size == 0:
            count_batch += 1
            t = timed_predict(model, batch)
            tot_time += t
            batch = []

        if count_sample > args.samples:
            break

    if len(batch) != 0:
        count_batch += 1
        t = timed_predict(model, batch)
        tot_time += t

    print("Using {} model, with batch size of {}, it took :\n".format(args.model, args.batch_size))
    print("{:.4f} s per batch\n".format(tot_time / count_batch))
    print("{:.4f} s per sample\n".format(tot_time / count_sample))
    print("(Average over {} samples)".format(args.samples))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark between HuggingFace BART and Pipeline BART. "
                                     "Need latest version of transformers (master branch)")
    parser.add_argument("--source", type=str)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--model", type=str, default=HF_MODEL, choices=[HF_MODEL, PIPELINE_MODEL])
    parser.add_argument('--fp32', dest='fp32', default=False, action='store_true')
    main(parser.parse_args())
