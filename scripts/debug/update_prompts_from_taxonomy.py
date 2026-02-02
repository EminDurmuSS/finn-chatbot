import json
from pathlib import Path

def update_prompts():
    # Paths
    json_path = Path("F:/finn-chatbot/data/category_taxonomy_v1.json")
    prompts_path = Path("F:/finn-chatbot/statement_copilot/core/prompts.py")
    
    # Load taxonomy
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Generate Rules String
    rules = []
    rules.append("### Categories (STRICT MAPPING)")
    rules.append("    You MUST map user terms to these exact IDs:")
    
    for cat_id, cat_data in data["categories"].items():
        subcats = cat_data.get("subcategories", {})
        
        # Parent mapping
        rules.append(f"\n    #### {cat_data['display_name']} (`{cat_id}`)")
        
        # Keywords for parent
        keywords = cat_data.get("keywords", [])[:5]
        if keywords:
            rules.append(f"    - Keywords: {', '.join(keywords)}")
            
        # Subcategories
        for sub_id, sub_data in subcats.items():
            sub_keywords = sub_data.get("keywords", [])
            sub_examples = sub_data.get("merchants_examples", [])
            
            # Combine triggers
            triggers = sub_keywords[:5] + sub_examples[:3]
            trigger_str = ", ".join(triggers)
            
            rules.append(f"    - `subcategories: [\"{sub_id}\"]` when user mentions: {trigger_str}")

    generated_rules = "\n".join(rules)

    # Read Prompt File
    with open(prompts_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Logic to find and replace the content
    # We look for "### Categories" and "### Direction"
    start_marker = "### Categories"
    end_marker = "### Direction"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("Error: Could not find markers in prompts.py")
        return

    # Keep the indentation/structure clean
    new_content = content[:start_idx] + generated_rules + "\n\n    " + content[end_idx:]
    
    # Write back
    with open(prompts_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("Successfully updated prompts.py with taxonomy rules.")

if __name__ == "__main__":
    update_prompts()
