import { TestBed } from '@angular/core/testing';

import { ApiStatusService } from './api-status.service';

describe('ApiStatusService', () => {
  let service: ApiStatusService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ApiStatusService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
