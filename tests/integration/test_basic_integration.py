"""Basic integration tests for the insurance risk analytics project."""

import pytest
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_project_structure():
    """Test that basic project structure exists."""
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    
    # Check key directories exist
    assert os.path.exists(os.path.join(project_root, 'src'))
    assert os.path.exists(os.path.join(project_root, 'tests'))
    assert os.path.exists(os.path.join(project_root, 'notebooks'))
    assert os.path.exists(os.path.join(project_root, 'data'))
    assert os.path.exists(os.path.join(project_root, 'docs'))
    assert os.path.exists(os.path.join(project_root, 'scripts'))
    assert os.path.exists(os.path.join(project_root, 'config'))


def test_src_imports():
    """Test that main source modules can be imported."""
    try:
        import src
        import src.core
        import src.models
        import src.utils
        import src.services
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import src modules: {e}")


def test_config_files_exist():
    """Test that important configuration files exist."""
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    
    # Check key files exist
    assert os.path.exists(os.path.join(project_root, 'requirements.txt'))
    assert os.path.exists(os.path.join(project_root, 'pyproject.toml'))
    assert os.path.exists(os.path.join(project_root, 'README.md'))
    assert os.path.exists(os.path.join(project_root, '.gitignore'))


def test_model_files_exist():
    """Test that model files exist in the correct location."""
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    models_dir = os.path.join(project_root, 'src', 'models')
    
    assert os.path.exists(os.path.join(models_dir, 'basic_modeling.py'))
    assert os.path.exists(os.path.join(models_dir, 'predictive_modeling_basic.py'))


def test_test_structure():
    """Test that test structure is properly organized."""
    tests_root = os.path.join(os.path.dirname(__file__), '..')
    
    assert os.path.exists(os.path.join(tests_root, 'unit'))
    assert os.path.exists(os.path.join(tests_root, 'integration'))
    assert os.path.exists(os.path.join(tests_root, 'unit', 'test_data_generation.py'))


if __name__ == "__main__":
    pytest.main([__file__]) 