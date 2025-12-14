import { Routes } from '@angular/router';
import { HomeComponent } from './features/home/home.component';
import { DetectComponent } from './features/detect/detect.component';
import { AboutComponent } from './features/about/about.component';

export const routes: Routes = [
  { path: '', component: HomeComponent, pathMatch: 'full' },
  { path: 'detect', component: DetectComponent },
  { path: 'about', component: AboutComponent },
  { path: '**', redirectTo: '' },
];
