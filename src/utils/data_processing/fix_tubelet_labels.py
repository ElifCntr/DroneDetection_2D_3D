import pandas as pd


def fix_tubelet_labels(csv_path, output_path=None):
    """
    Fix tubelet labels: if max_iou > 0, then label = 1
    """

    print(f"[INFO] Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"[INFO] Original data:")
    print(f"  Total tubelets: {len(df)}")
    print(f"  Positive labels (label=1): {sum(df['label'] == 1)}")
    print(f"  Negative labels (label=0): {sum(df['label'] == 0)}")
    print(f"  Tubelets with IoU > 0: {sum(df['max_iou'] > 0)}")
    print(f"  Tubelets with IoU = 0: {sum(df['max_iou'] == 0)}")
    print(f"  Negatives with IoU > 0: {sum((df['label'] == 0) & (df['max_iou'] > 0))}")

    # Show some examples of what will change
    negatives_with_iou = df[(df['label'] == 0) & (df['max_iou'] > 0)]
    if len(negatives_with_iou) > 0:
        print(f"\n[INFO] Examples of labels that will change (first 5):")
        for idx, row in negatives_with_iou.head().iterrows():
            print(f"  {row['filename']}: IoU={row['max_iou']:.3f} → will become positive")

    # Fix the labels
    print(f"\n[INFO] Applying fix: if max_iou > 0, then label = 1")

    # Count how many will change
    changes = sum((df['label'] == 0) & (df['max_iou'] > 0))

    # Apply the fix
    df.loc[df['max_iou'] > 0, 'label'] = 1

    print(f"[INFO] Fixed data:")
    print(f"  Total tubelets: {len(df)}")
    print(f"  Positive labels (label=1): {sum(df['label'] == 1)}")
    print(f"  Negative labels (label=0): {sum(df['label'] == 0)}")
    print(f"  Labels changed: {changes}")

    # Calculate percentages
    pos_pct = sum(df['label'] == 1) / len(df) * 100
    neg_pct = sum(df['label'] == 0) / len(df) * 100

    print(f"  New distribution: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative")

    # Save corrected CSV
    if output_path is None:
        output_path = csv_path.replace('.csv', '_corrected.csv')

    df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Corrected CSV saved to: {output_path}")

    return output_path, df


def verify_fix(csv_path):
    """Verify the fix worked correctly"""

    df = pd.read_csv(csv_path)

    print(f"[VERIFY] Checking fixed CSV...")

    # Should be no cases where IoU > 0 and label = 0
    bad_cases = sum((df['max_iou'] > 0) & (df['label'] == 0))

    if bad_cases == 0:
        print(f"✅ GOOD: All tubelets with IoU > 0 have label = 1")
    else:
        print(f"❌ ERROR: {bad_cases} tubelets still have IoU > 0 but label = 0")

    # All negatives should have IoU = 0
    negatives_with_iou = sum((df['label'] == 0) & (df['max_iou'] > 0))
    if negatives_with_iou == 0:
        print(f"✅ GOOD: All negative labels have IoU = 0")
    else:
        print(f"❌ ERROR: {negatives_with_iou} negative labels have IoU > 0")

    # Show IoU distribution for positives
    positives = df[df['label'] == 1]
    if len(positives) > 0:
        print(f"\n[INFO] IoU distribution for positive labels:")
        print(f"  Min IoU: {positives['max_iou'].min():.3f}")
        print(f"  Max IoU: {positives['max_iou'].max():.3f}")
        print(f"  Mean IoU: {positives['max_iou'].mean():.3f}")
        print(
            f"  IoU > 0.5: {sum(positives['max_iou'] > 0.5)} ({sum(positives['max_iou'] > 0.5) / len(positives) * 100:.1f}%)")
        print(
            f"  IoU ≤ 0.5: {sum(positives['max_iou'] <= 0.5)} ({sum(positives['max_iou'] <= 0.5) / len(positives) * 100:.1f}%)")


if __name__ == "__main__":

    # Your CSV path
    csv_path = "../../../data/tubelets/test_index.csv"

    print("=" * 60)
    print("FIXING TUBELET LABELS")
    print("=" * 60)

    try:
        # Fix the labels
        corrected_path, df_corrected = fix_tubelet_labels(csv_path)

        # Verify the fix
        print(f"\n" + "=" * 60)
        verify_fix(corrected_path)

        print(f"\n🎉 DONE!")
        print(f"  Original CSV: {csv_path}")
        print(f"  Corrected CSV: {corrected_path}")
        print(f"  Use the corrected CSV for model evaluation")

    except FileNotFoundError:
        print(f"❌ ERROR: CSV file not found: {csv_path}")
        print("Please check the path and try again")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()