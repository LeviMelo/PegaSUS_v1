from pegasus.registries.validators import validate_registry_tree

errors = validate_registry_tree("config/registries")
if errors:
    for e in errors:
        print(e)
    raise SystemExit(1)
print("registries valid")
