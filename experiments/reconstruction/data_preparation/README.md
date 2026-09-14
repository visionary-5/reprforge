# Energy input preparation

Public dataset: `vidore/vidore_v3_energy`, revision
`caec06d3c73434d635f710f93bcd898331c59f20`. The input manifest specifies ordered page identities for the official upgrade.
Download its `corpus/`, `queries/`, and `qrels/` Parquet files. Then:

```bash
python prepare_vidore_complete_task.py --corpus-dir /data/energy/corpus \
  --queries-dir /data/energy/queries --qrels-dir /data/energy/qrels \
  --output-root /data/new-energy-complete
python prepare_qrel_closed_slice.py --slice /data/new-energy-complete/slice.parquet \
  --route-root /data/new-energy-complete/route --query-count 16 \
  --target-pages 128 --seed 20260813 --output-root /data/new-energy-128
```

`energy-qrel-closed-128-manifest.json` records the historical selected indices.
CPU reconstruction was checked: the resulting 128 `(corpus_id, image_binary)`
rows equal the old slice exactly. However, the reconstructed Parquet container
has SHA-256 `172d2f8de0fa1a862faaff20d5be2c5e3e37f140a115d1a4c75661cb06b9a24c`,
whereas the original frozen file has SHA-256
`26020259fe345652806f2a1a2a8de08b08bc85739a32e8a7bfc40b0130be7502`.
Both report PyArrow 19.0.1 and Snappy. The cause of the serialization difference
is unconfirmed. The historical GPU runner intentionally retains its original
file-hash gate; these preparation scripts therefore recover the logical inputs,
and do not produce byte-identical Parquet containers. To run the same computation
on the reconstructed input, explicitly certify all 128 ordered IDs/image hashes:

```bash
python certify_energy_reconstruction.py --slice /data/new-energy-128/slice.parquet \
  --output-root /data/new-energy-input-certificate
```

This writes an input-equivalence receipt and a separate `gpu-protocol.json` with
the certified container hash, original protocol hash and original dataset hash.
Use that successor protocol with the official-upgrade GPU runner and prepared slice.
All model, batch, page, metric and timing definitions stay unchanged. Never edit
the original protocol or describe this preparation check as a GPU rerun. The new
independent endpoint uses fully pinned, verified preparations.
