"""
Analyze whether fraud cases show network structure that GNNs could exploit.

Answers:
  - Do fraud cases cluster around the same senders/receivers?
  - Do fraud senders also send clean transactions? (neighbor context)
  - Are there shared counterparties between fraud cases?
  - Is there any circular flow (sender↔receiver overlap)?

Usage:
    python scripts/fraud_network_analysis.py
    python scripts/fraud_network_analysis.py --variant v1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.data.prepare import prepare_data
from src.utils.config import load_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v1")
    args = parser.parse_args()

    p = prepare_data(load_variant(args.variant))
    df = p.df
    label_col = p.col_cfg["label"]
    fraud = df[df[label_col] == True]
    clean = df[df[label_col] == False]

    print(f"\n{'='*60}")
    print("FRAUD NETWORK ANALYSIS")
    print(f"{'='*60}")

    print(f"\n--- Basic counts ---")
    print(f"Total transactions:    {len(df):,}")
    print(f"Fraud transactions:    {len(fraud):,} ({100*len(fraud)/len(df):.3f}%)")
    print(f"Clean transactions:    {len(clean):,}")

    # --- Sender analysis ---
    print(f"\n--- Fraud sender analysis ---")
    fraud_senders = fraud["_sender"].value_counts()
    print(f"Unique fraud senders:         {len(fraud_senders):,}")
    print(f"Senders with 1 fraud txn:     {(fraud_senders == 1).sum():,}")
    print(f"Senders with 2+ fraud txns:   {(fraud_senders > 1).sum():,}")
    print(f"Senders with 5+ fraud txns:   {(fraud_senders >= 5).sum():,}")
    print(f"Max fraud txns per sender:    {fraud_senders.max()}")
    print(f"\nTop 10 fraud senders:")
    for acc, count in fraud_senders.head(10).items():
        total = len(df[df["_sender"] == acc])
        print(f"  {acc}: {count} fraud / {total} total ({100*count/total:.1f}%)")

    # --- Receiver analysis ---
    print(f"\n--- Fraud receiver analysis ---")
    fraud_receivers = fraud["_receiver"].value_counts()
    print(f"Unique fraud receivers:       {len(fraud_receivers):,}")
    print(f"Receivers with 2+ fraud txns: {(fraud_receivers > 1).sum():,}")
    print(f"Receivers with 5+ fraud txns: {(fraud_receivers >= 5).sum():,}")
    print(f"Max fraud txns per receiver:  {fraud_receivers.max()}")
    print(f"\nTop 10 fraud receivers:")
    for acc, count in fraud_receivers.head(10).items():
        total = len(df[df["_receiver"] == acc])
        print(f"  {acc}: {count} fraud / {total} total ({100*count/total:.1f}%)")

    # --- Neighbor context ---
    print(f"\n--- Neighbor context (can GNNs learn from neighborhood?) ---")
    fraud_sender_set = set(fraud["_sender"])
    fraud_receiver_set = set(fraud["_receiver"])

    clean_from_fraud_sender = clean[clean["_sender"].isin(fraud_sender_set)]
    print(f"Clean txns from fraud senders:   {len(clean_from_fraud_sender):,} ({100*len(clean_from_fraud_sender)/len(clean):.2f}% of clean)")

    clean_to_fraud_receiver = clean[clean["_receiver"].isin(fraud_receiver_set)]
    print(f"Clean txns to fraud receivers:   {len(clean_to_fraud_receiver):,} ({100*len(clean_to_fraud_receiver)/len(clean):.2f}% of clean)")

    # --- Overlap / circular flows ---
    print(f"\n--- Circular flows ---")
    overlap = fraud_sender_set & fraud_receiver_set
    print(f"Accounts that both send AND receive fraud: {len(overlap)}")
    if overlap:
        for acc in list(overlap)[:5]:
            sent = fraud[fraud["_sender"] == acc].shape[0]
            recv = fraud[fraud["_receiver"] == acc].shape[0]
            print(f"  {acc}: sent {sent}, received {recv}")

    # --- Shared counterparties ---
    print(f"\n--- Shared counterparties between fraud senders ---")
    fraud_sender_receivers = fraud.groupby("_sender")["_receiver"].apply(set)
    if len(fraud_sender_receivers) > 1:
        from itertools import combinations
        shared_count = 0
        pairs_checked = 0
        senders_list = list(fraud_sender_receivers.index)[:200]  # cap for speed
        for s1, s2 in combinations(senders_list, 2):
            pairs_checked += 1
            common = fraud_sender_receivers[s1] & fraud_sender_receivers[s2]
            if common:
                shared_count += 1
        print(f"Sender pairs sharing a fraud receiver: {shared_count} / {pairs_checked} checked")

    # --- 1-hop fraud density ---
    print(f"\n--- 1-hop fraud density ---")
    all_senders = set(df["_sender"])
    all_receivers = set(df["_receiver"])
    fraud_neighbor_senders = set()
    for recv in fraud_receiver_set:
        senders_to_recv = set(df[df["_receiver"] == recv]["_sender"])
        fraud_neighbor_senders |= senders_to_recv
    fraud_neighbors_who_fraud = fraud_neighbor_senders & fraud_sender_set
    print(f"Accounts sending to a fraud receiver:     {len(fraud_neighbor_senders):,}")
    print(f"Of those, also fraud senders themselves:   {len(fraud_neighbors_who_fraud):,}")
    if fraud_neighbor_senders:
        print(f"Fraud density in 1-hop neighborhood:      {100*len(fraud_neighbors_who_fraud)/len(fraud_neighbor_senders):.2f}%")
        print(f"Fraud density in full dataset:             {100*len(fraud_sender_set)/len(all_senders):.2f}%")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    repeat = (fraud_senders > 1).sum()
    print(f"Repeat fraud senders:    {repeat} / {len(fraud_senders)} ({100*repeat/len(fraud_senders):.1f}%)")
    print(f"Neighbor context:        {len(clean_from_fraud_sender):,} clean txns share a sender with fraud")
    print(f"Circular flows:          {len(overlap)} accounts")
    if fraud_neighbor_senders:
        density = 100 * len(fraud_neighbors_who_fraud) / len(fraud_neighbor_senders)
        baseline = 100 * len(fraud_sender_set) / len(all_senders)
        print(f"1-hop fraud enrichment:  {density:.2f}% vs {baseline:.2f}% baseline ({density/baseline:.1f}x)")
    print()


if __name__ == "__main__":
    main()
