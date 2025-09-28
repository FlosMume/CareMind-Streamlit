# Python script to compare requirements.txt and current.txt

def parse_requirements(file_path):
    """Parse a requirements-style file into a dictionary of package:version"""
    requirements = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                if '==' in line:
                    pkg, ver = line.split('==')
                    requirements[pkg.lower()] = ver
                else:
                    # Handle packages without version specifier
                    requirements[line.lower()] = None
    return requirements

# Load both files
requirements = parse_requirements('requirements.txt')
current = parse_requirements('current.txt')

# Check for missing or mismatched packages
missing = []
mismatched = []

for pkg, req_ver in requirements.items():
    if pkg not in current:
        missing.append(pkg)
    elif req_ver and current[pkg] != req_ver:
        mismatched.append((pkg, req_ver, current[pkg]))

# Print results
if not missing and not mismatched:
    print("✅ All dependencies from requirements.txt are present in current.txt with matching versions.")
else:
    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
    if mismatched:
        print("⚠️ Version mismatches:")
        for pkg, req_ver, cur_ver in mismatched:
            print(f"  - {pkg}: required {req_ver}, found {cur_ver}")

