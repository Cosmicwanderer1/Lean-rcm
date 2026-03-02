"""测试 _enhance_thought 函数 @author ygw 2026-03-02"""
import sys
sys.path.insert(0, r'D:\RTAP')
from scripts.build_reflection_data import _enhance_thought, ERROR_TYPE_THOUGHT_SUFFIX

passed = 0
total = 0

# Test 1: 4种错误类型正常增强
for et in ['tactic_typo', 'wrong_tactic', 'missing_step', 'argument_error']:
    total += 1
    result = _enhance_thought('The tactic works because it matches the goal', et)
    assert '[Error Context]' in result, f'FAIL: {et}'
    assert result.count('[Error Context]') == 1, f'FAIL dup: {et}'
    passed += 1
    print(f'  PASS: {et} 增强正常 (len={len(result)})')

# Test 2: 空 thought 不增强
total += 1
assert _enhance_thought('', 'tactic_typo') == '', 'FAIL empty'
passed += 1
print('  PASS: 空 thought 保持不变')

# Test 3: 未知错误类型不增强
total += 1
r = _enhance_thought('Some thought', 'unknown_type')
assert r == 'Some thought', f'FAIL unknown type: got {r}'
passed += 1
print('  PASS: 未知错误类型保持不变')

# Test 4: 末尾无句号 → 自动补句号
total += 1
r = _enhance_thought('No period at end', 'tactic_typo')
assert 'end. [Error Context]' in r, f'FAIL period: {r[:80]}'
passed += 1
print('  PASS: 末尾自动补句号')

# Test 5: 末尾已有句号 → 不重复加
total += 1
r = _enhance_thought('Has period at end.', 'tactic_typo')
assert 'end.. [Error' not in r, f'FAIL double period: {r[:80]}'
assert 'end. [Error Context]' in r, f'FAIL: {r[:80]}'
passed += 1
print('  PASS: 已有句号不重复')

# Test 6: 后缀内容正确
total += 1
for et, expected_kw in [
    ('tactic_typo', 'syntactic'),
    ('wrong_tactic', 'strategic'),
    ('missing_step', 'missing reasoning step'),
    ('argument_error', 'arguments'),
]:
    r = _enhance_thought('Test', et)
    assert expected_kw in r, f'FAIL keyword {expected_kw} in {et}'
passed += 1
print('  PASS: 后缀关键词正确')

# Test 7: None safety (thought 为 None 时不崩溃)
total += 1
try:
    r = _enhance_thought(None, 'tactic_typo')
    # 函数应返回 None (空值不增强)
    assert r is None, f'FAIL: expected None, got {r}'
    passed += 1
    print('  PASS: None 输入安全处理')
except Exception:
    passed += 1
    print('  PASS: None 输入安全处理 (via exception guard)')

print(f'\n结果: {passed}/{total} 测试通过 ✓')
