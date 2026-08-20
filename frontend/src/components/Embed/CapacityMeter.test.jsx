import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import CapacityMeter from './CapacityMeter.jsx';

describe('CapacityMeter', () => {
  it('shows an idle hint when there is no data and not loading', () => {
    render(<CapacityMeter data={null} loading={false} messageBytes={0} />);
    expect(screen.getByText(/add an image/i)).toBeInTheDocument();
  });

  it('shows an "estimating" hint while loading with no data yet', () => {
    render(<CapacityMeter data={null} loading={true} messageBytes={0} />);
    expect(screen.getByText(/estimating capacity/i)).toBeInTheDocument();
  });

  it('shows the non-recoverable message for adaptive/steganogan-style data', () => {
    render(
      <CapacityMeter
        data={{ recoverable: false, max_message_bytes: 0 }}
        loading={false}
        messageBytes={0}
      />
    );
    expect(screen.getByText(/statistical noise/i)).toBeInTheDocument();
    expect(screen.queryByText(/bytes$/i)).not.toBeInTheDocument();
  });

  it('shows used/max bytes for a recoverable method within capacity', () => {
    render(
      <CapacityMeter
        data={{ recoverable: true, max_message_bytes: 1000 }}
        loading={false}
        messageBytes={200}
      />
    );
    expect(screen.getByText('200 / 1,000 bytes')).toBeInTheDocument();
    expect(screen.queryByText(/too large/i)).not.toBeInTheDocument();
  });

  it('flags an over-capacity message with an error and no crash on the bar width', () => {
    render(
      <CapacityMeter
        data={{ recoverable: true, max_message_bytes: 100 }}
        loading={false}
        messageBytes={500}
      />
    );
    expect(screen.getByText('500 / 100 bytes')).toBeInTheDocument();
    expect(screen.getByText(/too large/i)).toBeInTheDocument();
  });

  it('treats a missing messageBytes as zero used', () => {
    render(
      <CapacityMeter data={{ recoverable: true, max_message_bytes: 50 }} loading={false} />
    );
    expect(screen.getByText('0 / 50 bytes')).toBeInTheDocument();
  });
});
