from pathlib import Path
import argparse
import json

# TODO: Test!!!


def join_coco(base_dataset_path: Path, join_dataset_path: Path, output: Path,
    override: bool, increment: bool):

    assert not (override and increment)
    assert (base_dataset_path != output) and (join_dataset_path != output)

    # Read datasets
    base_dataset = json.loads(base_dataset_path.read_text())
    join_dataset = json.loads(join_dataset_path.read_text())

    b_images = base_dataset["images"]
    b_annots = base_dataset["annotations"]
    b_cats = base_dataset["categories"]

    j_images = join_dataset["images"]
    j_annots = join_dataset["annotations"]
    j_cats = join_dataset["categories"]
    
    # Join categories
    cat_id_base = [i["id"] for i in b_cats]
    cat_name_base = [i["name"] for i in b_cats]
    last_id = max(cat_id_base) + 1
    mapped_cat_ids = {}
    for cat in j_cats:
        if cat["name"] in cat_name_base:
            dup_idx = cat_name_base.index(cat["name"])
            mapped_cat_ids[cat["id"]] = b_cats[dup_idx]["id"]
            continue

        if cat["id"] in cat_id_base:
            mapped_cat_ids[cat["id"]] = last_id
            cat["id"] = last_id
            last_id += 1

        b_cats.append(cat)

    # Join images
    img_id_base = [i["id"] for i in b_images]
    img_name_base = [i["file_name"] for i in b_images]
    last_id = max(img_id_base) + 1
    mapped_img_ids = {}
    for img in j_images:
        if img["file_name"] in img_name_base:
            dup_idx = img_name_base.index(img["file_name"])
            mapped_img_ids[img["id"]] = b_images[dup_idx]["id"]
            continue

        if img["id"] in img_id_base:
            if increment:
                mapped_img_ids[img["id"]] = last_id
                img["id"] = last_id
                last_id += 1
                b_images.append(img)
            else:
                raise KeyError("Image ID already found on base dataset")
        else:
            b_images.append(img)

    # Join annotations
    ann_id_base = [i["id"] for i in b_annots]
    last_id = max(ann_id_base) + 1
    for ann in j_annots:
        if ann["image_id"] in mapped_img_ids:
            ann["image_id"] = mapped_img_ids[ann["image_id"]]
        
        if ann["category_id"] in mapped_cat_ids:
            ann["category_id"] = mapped_cat_ids[ann["category_id"]]

        if ann["id"] in ann_id_base:
            dup_idx = ann_id_base.index(ann["id"])

            if ann == b_annots[dup_idx]:
                continue

            if override:
                del(b_annots[dup_idx])
                b_annots.append(ann)
            elif increment:
                ann["id"] = last_id
                b_annots.append(ann)
                last_id += 1
            else:
                raise KeyError("Annotation ID already found on base dataset")
        else:
            b_annots.append(ann)
    
    
    output.write_text(json.dumps(base_dataset))



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Join two coco datasets")
    
    argparser.add_argument(
        'base_dataset',
        help="Path to the json base dataset",
        type=Path
    )
    argparser.add_argument(
        'join_dataset',
        help="Path to the second json dataset",
        type=Path
    )
    argparser.add_argument(
        'output',
        help="Output json path",
        type=Path
    )
    argparser.add_argument(
        '--override',
        help="Overrides the base dataset annotations if it has duplicated ids",
        action="store_true"
    )
    argparser.add_argument(
        '--increment',
        help="Increments the annotations and images id if they are duplicated",
        action="store_true"
    )
    
    args = argparser.parse_args()

    join_coco(args.base_dataset, args.join_dataset, args.output, args.override,
        args.increment)
