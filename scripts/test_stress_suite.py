
import json
import logging
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import argparse

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Setup logging - silence internal logs
logging.basicConfig(level=logging.ERROR)
console = Console()

try:
    from statement_copilot.workflow import StatementCopilot
except ImportError:
    console.print("[red]Error: Could not import StatementCopilot. check python path.[/red]")
    sys.exit(1)

def load_suite(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_tests():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, help="Run specific question ID")
    args = parser.parse_args()

    data_path = PROJECT_ROOT / "data" / "stress_test_suite.json"
    if not data_path.exists():
        console.print(f"[red]Test suite not found at {data_path}[/red]")
        return

    suite = load_suite(data_path)
    
    # Filter if ID provided
    if args.id:
        suite = [x for x in suite if x["id"] == args.id]
        if not suite:
            console.print(f"[red]Question {args.id} not found[/red]")
            return

    copilot = StatementCopilot()
    
    results = []
    
    table = Table(title="Stress Test Results")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Question", style="white")
    table.add_column("Pass/Fail", style="bold")
    table.add_column("Missing Keywords", style="red")

    passed_count = 0
    
    for item in suite:
        q_id = item["id"]
        question = item["question"]
        expected = item.get("expected_keywords", [])
        
        console.print(f"\n[bold blue]Running Q{q_id}:[/bold blue] {question}")
        
        try:
            # Run copilot
            response = copilot.chat(
                message=question,
                session_id=f"stress_test_{q_id}",
                tenant_id="default_tenant",
                user_id="default_user"
            )
            
            final_answer = response.get("final_answer", "")
            
            # Verification
            missing = []
            for keyword in expected:
                if keyword.lower() not in final_answer.lower():
                    missing.append(keyword)
            
            is_passed = len(missing) == 0
            if is_passed:
                passed_count += 1
                status = "[green]PASS[/green]"
            else:
                status = "[red]FAIL[/red]"
            
            table.add_row(str(q_id), question[:50]+"...", status, ", ".join(missing))
            
            # Print detail for failure
            if not is_passed:
                console.print(Panel(final_answer, title=f"Result Q{q_id} (Failed)", border_style="red"))
                console.print(f"[dim]Expected to find: {expected}[/dim]")
            else:
                console.print(f"[green]>> Passed[/green]")

        except Exception as e:
            console.print(f"[red]Error execution Q{q_id}: {e}[/red]")
            table.add_row(str(q_id), question[:50]+"...", "[red]ERROR[/red]", str(e))

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold]Total Passed: {passed_count}/{len(suite)}[/bold]")

if __name__ == "__main__":
    run_tests()
