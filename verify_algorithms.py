#!/usr/bin/env python3
"""Verify algorithm complexity analysis is correct"""
import sys
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.complexity import ComplexityAnalyzer

# Read actual test pseudocodes
test_dir = Path("tests/pseudocode")
algorithms = [
    ("binary_search.txt", "O(log n)", "Ω(1)", "Θ(log n)"),
    ("constant.txt", "O(1)", "Ω(1)", "Θ(1)"),
    ("factorial.txt", "O(n)", "Ω(n)", "Θ(n)"),
    ("for_simple.txt", "O(n)", "Ω(n)", "Θ(n)"),
    ("nested_for.txt", "O(n²)", "Ω(n²)", "Θ(n²)"),
]

analyzer = ComplexityAnalyzer()

print("="*80)
print(" ALGORITHM COMPLEXITY VERIFICATION")
print("="*80)

results = []

for algo_file, expected_o, expected_omega, expected_theta in algorithms:
    path = test_dir / algo_file
    if not path.exists():
        print(f"\n{algo_file}: NOT FOUND")
        continue
    
    code = path.read_text()
    name = algo_file.replace(".txt", "").replace("_", " ").title()
    
    print(f"\n{name}:")
    
    # Analyze
    result = analyzer.analyze(code)
    
    # Check if matches expected
    o_match = result.big_o == expected_o
    omega_match = result.omega == expected_omega
    theta_match = result.theta == expected_theta
    
    status = "✓" if (o_match and omega_match and theta_match) else "✗"
    print(f"  {status} Big-O:  {result.big_o:15} (expected: {expected_o})")
    print(f"  {status} Omega:  {result.omega:15} (expected: {expected_omega})")
    print(f"  {status} Theta:  {result.theta:15} (expected: {expected_theta})")
    
    results.append({
        "name": name,
        "all_match": o_match and omega_match and theta_match,
        "o_match": o_match,
        "omega_match": omega_match,
        "theta_match": theta_match
    })

print("\n" + "="*80)
print(" SUMMARY")
print("="*80)

for r in results:
    overall = "✓ PASS" if r['all_match'] else "✗ FAIL"
    print(f"{r['name']:20} {overall:10} (O:{str(r['o_match']):5} Ω:{str(r['omega_match']):5} Θ:{str(r['theta_match']):5})")

all_pass = all(r['all_match'] for r in results)
print(f"\nAll algorithms correct: {'YES ✓' if all_pass else 'NO ✗'}")
