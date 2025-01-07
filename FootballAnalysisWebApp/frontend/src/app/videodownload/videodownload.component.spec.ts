import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VideodownloadComponent } from './videodownload.component';

describe('VideodownloadComponent', () => {
  let component: VideodownloadComponent;
  let fixture: ComponentFixture<VideodownloadComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ VideodownloadComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(VideodownloadComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
