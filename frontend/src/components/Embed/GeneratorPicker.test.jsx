import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import GeneratorPicker from './GeneratorPicker.jsx';
import { METHOD_ORDER, METHODS } from '../../config/methods.js';

describe('GeneratorPicker', () => {
  it('renders one radio button per method, in METHOD_ORDER', () => {
    render(<GeneratorPicker value="lsb" onChange={() => {}} />);
    const radios = screen.getAllByRole('radio');
    expect(radios).toHaveLength(METHOD_ORDER.length);
  });

  it('marks the selected method as checked and the rest as unchecked', () => {
    render(<GeneratorPicker value="dct" onChange={() => {}} />);
    expect(screen.getByRole('radio', { name: new RegExp(METHODS.dct.plain) })).toHaveAttribute(
      'aria-checked', 'true'
    );
    expect(screen.getByRole('radio', { name: new RegExp(METHODS.lsb.plain) })).toHaveAttribute(
      'aria-checked', 'false'
    );
  });

  it('calls onChange with the clicked method id', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<GeneratorPicker value="lsb" onChange={onChange} />);

    await user.click(screen.getByRole('radio', { name: new RegExp(METHODS.steganogan.plain) }));

    expect(onChange).toHaveBeenCalledWith('steganogan');
  });

  it('tags non-recoverable methods "noise only" and leaves recoverable ones untagged', () => {
    render(<GeneratorPicker value="lsb" onChange={() => {}} />);
    const adaptiveCard = screen.getByRole('radio', { name: new RegExp(METHODS.adaptive.plain) });
    const lsbCard = screen.getByRole('radio', { name: new RegExp(METHODS.lsb.plain) });

    expect(adaptiveCard).toHaveTextContent('noise only');
    expect(lsbCard).not.toHaveTextContent('noise only');
  });

  it('shows the technical name alongside the plain-English label', () => {
    render(<GeneratorPicker value="lsb" onChange={() => {}} />);
    const card = screen.getByRole('radio', { name: new RegExp(METHODS.lsb.plain) });
    expect(card).toHaveTextContent(METHODS.lsb.name);
  });
});
