import json
from pathlib import Path

def generate_prompt_rules():
    # Load taxonomy
    json_path = Path("F:/finn-chatbot/data/category_taxonomy_v1.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data["categories"]
    
    rules = []
    rules.append("### Categories (STRICT MAPPING)")
    rules.append("You MUST mapping user terms to these exact IDs:")
    
    for cat_id, cat_data in categories.items():
        subcats = cat_data.get("subcategories", {})
        
        # Parent mapping
        rules.append(f"\n#### {cat_data['display_name']} (`{cat_id}`)")
        
        # Keywords for parent
        keywords = cat_data.get("keywords", [])[:5]
        if keywords:
            rules.append(f"- Keywords: {', '.join(keywords)}")
            
        # Subcategories
        for sub_id, sub_data in subcats.items():
            sub_keywords = sub_data.get("keywords", [])
            sub_examples = sub_data.get("merchants_examples", [])
            
            # Combine triggers
            triggers = sub_keywords[:4] + sub_examples[:3]
            trigger_str = ", ".join(triggers)
            
            rules.append(f"- `subcategories: [\"{sub_id}\"]` when user mentions: {trigger_str}")

    print("\n".join(rules))

if __name__ == "__main__":
    generate_prompt_rules()
