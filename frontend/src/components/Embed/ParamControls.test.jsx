import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ParamControls from './ParamControls.jsx';
import { defaultParams } from '../../config/methods.js';

describe('ParamControls', () => {
  it('renders a field for every param of the method', () => {
    render(
      <ParamControls methodId="dct" values={defaultParams('dct')} onChange={() => {}} />
    );
    expect(screen.getByLabelText('Coefficient band')).toBeInTheDocument();
    expect(screen.getByLabelText('Strength (quant. step)')).toBeInTheDocument();
  });

  it('renders nothing for a method with no configurable params', () => {
    const { container } = render(
      <ParamControls methodId="steganogan" values={{}} onChange={() => {}} />
    );
    expect(container.querySelectorAll('input, select')).toHaveLength(0);
  });

  it('hides a param whose showIf condition is not met (lsb step, strategy=sequential)', () => {
    render(
      <ParamControls
        methodId="lsb"
        values={{ strategy: 'sequential', step: 1, bit_depth: 1 }}
        onChange={() => {}}
      />
    );
    expect(screen.queryByLabelText('Skip step (N)')).not.toBeInTheDocument();
  });

  it('shows the param once its showIf condition is met (lsb step, strategy=skip)', () => {
    render(
      <ParamControls
        methodId="lsb"
        values={{ strategy: 'skip', step: 7, bit_depth: 1 }}
        onChange={() => {}}
      />
    );
    expect(screen.getByLabelText('Skip step (N)')).toBeInTheDocument();
  });

  it('fires onChange with the raw string value for a select field', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <ParamControls methodId="lsb" values={defaultParams('lsb')} onChange={onChange} />
    );

    await user.selectOptions(screen.getByLabelText('Pixel selection'), 'skip');

    expect(onChange).toHaveBeenCalledWith('strategy', 'skip');
  });

  it('fires onChange with a parsed Number (not a string) for a number field', () => {
    const onChange = vi.fn();
    render(
      <ParamControls methodId="lsb" values={defaultParams('lsb')} onChange={onChange} />
    );

    fireEvent.change(screen.getByLabelText('Bits per pixel'), { target: { value: '3' } });

    expect(onChange).toHaveBeenCalledWith('bit_depth', 3);
    expect(typeof onChange.mock.calls[0][1]).toBe('number');
  });

  it('emits an empty string (not NaN) when a number field is cleared', () => {
    const onChange = vi.fn();
    render(
      <ParamControls methodId="lsb" values={defaultParams('lsb')} onChange={onChange} />
    );
    fireEvent.change(screen.getByLabelText('Bits per pixel'), { target: { value: '' } });
    expect(onChange).toHaveBeenCalledWith('bit_depth', '');
  });
});
